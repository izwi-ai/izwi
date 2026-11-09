mod decode;
mod device_decode;
pub(crate) mod metal_kernels;
mod nemo;
mod physical;
mod preprocessor;

use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};

use crate::backends::state::{
    InvocationTensorComponentValue, InvocationTensorUpdateV2, PhysicalStateTransactionId,
    StateComponentValue, TensorStateArena,
};
use crate::backends::{BackendKind, DTypeSelection, DeviceProfile};
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::kernels::cuda::shape_graphs::{ShapeGraphLimits, ShapeGraphs};
use crate::kernels::depthwise_conv::{depthwise_conv1d, DepthwiseConv1dSpec};
use crate::kernels::f16_weight_linear::{
    load_weight_only_f16, F16WeightLinear, PointwiseConv1d, ProjectionLinear, WeightOnlyF16Load,
    F16_PROJECTION_MAX_RELATIVE_L2,
};
use crate::kernels::numeric_check::{
    run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};
use crate::kernels::relpos_attention;
use crate::kernels::transducer_greedy::{tdt_step, TdtRow, TdtRules, TdtStep};
use crate::kv::v2::{
    ComponentShapeInstantiation, DomainStepIntent, ShapeAxis, ShapeDimensionValue,
    StateComponentId, StateUpdateKind,
};
use crate::model::ModelVariant;
use crate::models::shared::detokenize::IncrementalDetokenizer;
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::memory::exact::exact_zeros;
use crate::models::shared::weights::mlx;
use crate::tokenizer::Tokenizer;

use decode::decode_tokens;
use nemo::{ensure_parakeet_artifacts, ParakeetArtifacts};
pub(crate) use nemo::{parakeet_checkpoint_paths, parakeet_native_artifacts_complete};
#[cfg(test)]
pub(crate) use physical::{parakeet_physical_state_spec, PARAKEET_PREDICTOR_STATE_DOMAIN};
pub(crate) use physical::{ParakeetPhysicalStateSpec, PARAKEET_PREDICTOR_STATE_GROUP};
use preprocessor::{ParakeetLogMel, ParakeetPreprocessor};

/// The dtype the Parakeet loader reads every weight in, and the dtype of the
/// encoder's residual stream on every backend. The encoder's projection
/// weights are then stored in F16 on Metal or BF16 on CUDA (see
/// [`resolve_parakeet_projection_precision`]). Load admission prices the
/// checkpoint from both (`runtime/lifecycle/asr_memory.rs`).
pub(crate) const PARAKEET_WEIGHT_DTYPE: DType = DType::F32;

/// Parakeet's dtype override. On Metal, `f32` keeps every weight in F32 and
/// `f16` stores the encoder projections in F16; on CUDA, `f32` keeps every
/// weight in F32 and `bf16` (the default) runs the projections in BF16; see
/// [`resolve_parakeet_projection_precision`].
pub(crate) const PARAKEET_ASR_DTYPE_ENV: &str = "IZWI_PARAKEET_ASR_DTYPE";

/// The precision of the encoder's projections (FFN, attention q/k/v/out/pos,
/// pointwise convs and the subsampling output projection) on `device`:
/// - F16 on Metal: weight-only F16 with F32 activations, as in whisper.cpp's
///   Parakeet converter. A failed load-time self-check still falls back to
///   F32 ([`load_weight_only_f16`]).
/// - BF16 on CUDA ([`parakeet_cuda_projection_precision`]): BF16 weights and
///   GEMMs, each encoder block casting its input to BF16 once and its output
///   back to the F32 residual stream.
/// - F32 otherwise: every weight stays F32.
///
/// Subsampling convs, depthwise conv and BatchNorm, LayerNorms, positional
/// biases, the relative-position attention core, and the TDT predictor and
/// joint stay F32 in every tier.
pub(crate) fn resolve_parakeet_projection_precision(device: &DeviceProfile) -> DTypeSelection {
    let family = std::env::var(PARAKEET_ASR_DTYPE_ENV).ok();
    if device.kind.is_cuda() {
        return parakeet_cuda_projection_precision(
            device.capabilities.supports_bf16,
            family.as_deref(),
        );
    }
    device.resolve_metal_weight_only_f16(
        family
            .as_deref()
            .map(|value| (PARAKEET_ASR_DTYPE_ENV, value)),
    )
}

/// The CUDA projection tier: BF16 on a device that supports it, unless the
/// family override `raw` names F32. The F32 encoder is launch-bound on an
/// H100 (122 ms per 10 s clip); BF16 keeps F32's range, so it needs no
/// load-time range probe.
fn parakeet_cuda_projection_precision(supports_bf16: bool, raw: Option<&str>) -> DTypeSelection {
    let raw = raw.map(str::trim).filter(|raw| !raw.is_empty());
    let note = match raw {
        Some(raw) => match crate::backends::parse_dtype_name(raw) {
            Some(DType::F32) => {
                return DTypeSelection {
                    dtype: DType::F32,
                    reason: format!("{PARAKEET_ASR_DTYPE_ENV}={raw} keeps every weight in F32")
                        .into(),
                };
            }
            Some(DType::BF16) => format!("; {PARAKEET_ASR_DTYPE_ENV}={raw}"),
            _ => format!(
                "; {PARAKEET_ASR_DTYPE_ENV}={raw} is not a CUDA Parakeet precision and is ignored"
            ),
        },
        None => String::new(),
    };
    if !supports_bf16 {
        return DTypeSelection {
            dtype: DType::F32,
            reason: format!("the CUDA device reports no BF16 support; weights stay F32{note}")
                .into(),
        };
    }
    DTypeSelection {
        dtype: DType::BF16,
        reason: format!(
            "CUDA default: BF16 encoder projections and GEMMs with an F32 residual stream{note}"
        )
        .into(),
    }
}

/// Whether checkpoint tensor `name` is one of the encoder projections that
/// load in F16 when [`resolve_parakeet_projection_precision`] selects F16
/// (their biases stay F32). Admission prices these at F16.
pub(crate) fn parakeet_tensor_is_f16_projection(name: &str) -> bool {
    let Some(encoder) = name.strip_prefix("encoder.") else {
        return false;
    };
    if encoder == "pre_encode.out.weight" {
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
            "feed_forward1.linear1.weight"
                | "feed_forward1.linear2.weight"
                | "feed_forward2.linear1.weight"
                | "feed_forward2.linear2.weight"
                | "self_attn.linear_q.weight"
                | "self_attn.linear_k.weight"
                | "self_attn.linear_v.weight"
                | "self_attn.linear_out.weight"
                | "self_attn.linear_pos.weight"
                | "conv.pointwise_conv1.weight"
                | "conv.pointwise_conv2.weight"
        )
}

/// Whether checkpoint tensor `name` loads in BF16 when
/// [`resolve_parakeet_projection_precision`] selects BF16: the F16 set plus
/// the subsampling output projection's bias, which its BF16 GEMM adds.
/// Admission prices these at BF16.
pub(crate) fn parakeet_tensor_is_bf16_projection(name: &str) -> bool {
    name == "encoder.pre_encode.out.bias" || parakeet_tensor_is_f16_projection(name)
}

/// How the encoder's projections load: the tier
/// [`resolve_parakeet_projection_precision`] selects, on its device.
#[derive(Debug, Clone, Copy)]
enum Projections<'a> {
    /// Every weight dense F32 on the checkpoint's device.
    F32,
    /// Weight-only F16 on the device: F32 activations.
    F16(&'a Device),
    /// BF16 weights and GEMMs on the device: each block casts its input to
    /// BF16 once and its output back to F32.
    Bf16(&'a Device),
}

impl<'a> Projections<'a> {
    /// The builder projections read from: the host for a converted tier, so
    /// the weight is converted there and its F32 form never reaches the GPU.
    fn vb<'v>(self, vb: &VarBuilder<'v>) -> VarBuilder<'v> {
        match self {
            Self::F32 => vb.clone(),
            Self::F16(_) | Self::Bf16(_) => vb.clone().set_device(Device::Cpu),
        }
    }

    fn f16_device(self) -> Option<&'a Device> {
        match self {
            Self::F16(device) => Some(device),
            Self::F32 | Self::Bf16(_) => None,
        }
    }

    /// The dtype the projection GEMMs run in.
    fn dtype(self) -> DType {
        match self {
            Self::Bf16(_) => DType::BF16,
            Self::F32 | Self::F16(_) => DType::F32,
        }
    }

    /// `linear`, read through [`Self::vb`], in this tier.
    fn linear(self, linear: Linear) -> Result<ProjectionLinear> {
        match self {
            Self::Bf16(device) => Ok(ProjectionLinear::Dense(bf16_linear(
                linear.weight(),
                linear.bias(),
                device,
            )?)),
            Self::F32 | Self::F16(_) => {
                ProjectionLinear::new(linear, self.f16_device()).map_err(Error::from)
            }
        }
    }

    /// The pointwise conv `conv`, read through [`Self::vb`], in this tier.
    fn pointwise(self, conv: Conv1d) -> Result<ConvPointwise> {
        match self {
            Self::Bf16(device) => {
                let (out_channels, in_channels, _) = conv.weight().dims3()?;
                Ok(ConvPointwise::Bf16(bf16_linear(
                    &conv.weight().reshape((out_channels, in_channels))?,
                    conv.bias(),
                    device,
                )?))
            }
            Self::F32 | Self::F16(_) => Ok(ConvPointwise::Conv(PointwiseConv1d::new(
                conv,
                self.f16_device(),
            )?)),
        }
    }
}

/// A BF16 copy of a host weight (and bias) on `device`, converted on the host
/// and uploaded once at its exact size.
fn bf16_linear(weight: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Linear> {
    let upload = |tensor: &Tensor| tensor.to_dtype(DType::BF16)?.to_device(device);
    Ok(Linear::new(upload(weight)?, bias.map(upload).transpose()?))
}

/// `x` as a contiguous tensor of `dtype`. A cast writes contiguous output, so
/// a strided view costs one kernel either way.
fn contiguous_as(x: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    if x.dtype() == dtype {
        x.contiguous()
    } else {
        x.to_dtype(dtype)
    }
}

const SAMPLE_RATE: u32 = 16_000;
const PRED_HIDDEN: usize = 640;
const SUBSAMPLING_FACTOR: usize = 8;
const FRAME_HOP_MS: f32 = 10.0;
const DEFAULT_MAX_SYMBOLS: usize = 10;
/// Seconds per encoder frame: 10 ms feature hop × 8× subsampling.
const ENCODER_FRAME_SECONDS: f64 = FRAME_HOP_MS as f64 * SUBSAMPLING_FACTOR as f64 / 1000.0;
pub(crate) const PARAKEET_RETAINED_PREFILL_STAGE: &str = "asr.prefill.parakeet.predictor";
pub(crate) const PARAKEET_RETAINED_DECODE_STAGE: &str = "asr.decode.parakeet.joint";
pub(crate) const PARAKEET_RETAINED_WORKSPACE_PER_ROW_BYTES: u64 = 16 * 1024 * 1024;

/// FastConformer encoder geometry. `V3` is parakeet-tdt-0.6b-v3; tests build
/// tiny random encoders from smaller values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncoderGeometry {
    mel_bins: usize,
    subsampling_channels: usize,
    layers: usize,
    dim: usize,
    heads: usize,
    ff_dim: usize,
    conv_kernel: usize,
}

impl EncoderGeometry {
    const V3: Self = Self {
        mel_bins: 128,
        subsampling_channels: 256,
        layers: 24,
        dim: 1024,
        heads: 8,
        ff_dim: 4096,
        conv_kernel: 9,
    };

    const fn head_dim(self) -> usize {
        self.dim / self.heads
    }

    /// Mel bins left after the three stride-2 subsampling convolutions.
    fn subsampled_mel_bins(self) -> usize {
        subsampled_len_3x(self.mel_bins)
    }
}

/// Long audio is encoded in overlapping windows of encoder frames instead of
/// one full-attention pass whose memory grows as T². Each window keeps only
/// its centre, so every kept frame sees `left_context` frames before it and
/// `right_context` after it unless it lies at the real start or end of the
/// audio. One TDT decode then runs over the stitched frames, which keeps the
/// predictor state and frame timestamps continuous across seams. This is
/// NVIDIA's chunked-inference recipe for this model with simulated
/// streaming (left/chunk/right buffers, one decode over concatenated outputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncoderWindowing {
    frames: usize,
    left_context: usize,
    right_context: usize,
}

impl EncoderWindowing {
    /// 81.92 s windows with 10.24 s of context on each side, keeping 61.44 s.
    /// A power-of-two window keeps Metal's pooled buffers exactly sized, and
    /// 128 frames of context exceed the 96 frames over which 24 layers of
    /// kernel-9 convolutions spread window-edge effects.
    const V3: Self = Self {
        frames: 1024,
        left_context: 128,
        right_context: 128,
    };
}

/// One encoder window in encoder frames: `[start, end)` is encoded and
/// `[keep_start, keep_end)` is copied into the stitched output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncoderWindow {
    start: usize,
    end: usize,
    keep_start: usize,
    keep_end: usize,
}

impl EncoderWindow {
    /// The one window of an input of at most one window: every frame is
    /// encoded and kept.
    const fn whole(encoded_len: usize) -> Self {
        Self {
            start: 0,
            end: encoded_len,
            keep_start: 0,
            keep_end: encoded_len,
        }
    }

    /// The log-mel columns that produce this window's frames: whole groups
    /// of the ×8 subsampling, clipped to the spectrogram. Stopping at
    /// `8 · end` also drops the STFT padding column when the valid frames are
    /// a multiple of 8, where it would add an encoder frame NeMo masks.
    fn feature_columns(&self, feature_frames: usize) -> Range<usize> {
        self.start * SUBSAMPLING_FACTOR..(self.end * SUBSAMPLING_FACTOR).min(feature_frames)
    }
}

/// Plans windows of exactly `window` frames over `encoded_len` frames. The
/// first window keeps `[0, window - right)`; each later one starts `left`
/// frames before the previous kept range ends, and the last is shifted left
/// to end at the audio's end. Inputs of at most `window` frames are a single
/// window, which is the whole-utterance encode.
fn plan_encoder_windows(
    encoded_len: usize,
    window: usize,
    left: usize,
    right: usize,
) -> Result<Vec<EncoderWindow>> {
    if encoded_len <= window {
        return Ok((encoded_len > 0)
            .then_some(EncoderWindow::whole(encoded_len))
            .into_iter()
            .collect());
    }
    if window <= left.saturating_add(right) {
        return Err(Error::InvalidInput(format!(
            "Parakeet encoder window {window} leaves no centre between contexts {left} and {right}"
        )));
    }
    let mut windows = Vec::with_capacity(encoded_len.div_ceil(window - left - right) + 1);
    let mut keep_start = 0usize;
    loop {
        let start = keep_start.saturating_sub(left);
        let end = start + window;
        if end >= encoded_len {
            windows.push(EncoderWindow {
                start: encoded_len - window,
                end: encoded_len,
                keep_start,
                keep_end: encoded_len,
            });
            return Ok(windows);
        }
        let keep_end = end - right;
        windows.push(EncoderWindow {
            start,
            end,
            keep_start,
            keep_end,
        });
        keep_start = keep_end;
    }
}

/// Windows [`plan_encoder_windows`] plans over `encoded_len` frames.
fn encoder_window_count(encoded_len: usize, windowing: EncoderWindowing) -> Result<usize> {
    Ok(plan_encoder_windows(
        encoded_len,
        windowing.frames,
        windowing.left_context,
        windowing.right_context,
    )?
    .len()
    .max(1))
}

/// Kill switch for stacking a long input's encoder windows as batch rows (a
/// boolean; on when unset). Off, the windows encode one after another.
/// Read once: admission prices the stacked rows from it, and the encoder
/// runs them.
pub(crate) const PARAKEET_WINDOW_BATCHING_ENV: &str = "IZWI_PARAKEET_WINDOW_BATCHING";

fn window_batching_enabled() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(PARAKEET_WINDOW_BATCHING_ENV, true));
    ON.get()
}

/// The most windows of one long input the encoder runs as the rows of one
/// batch on `backend`. Every window of a long input has the same frames, so
/// the rows need no padding and a batch costs about one window's launches,
/// which is what a launch-bound GPU pays for. A full 1,024-frame window's
/// layer buffers are about 416 MiB, so CUDA stacks four (about 1.4 GiB).
/// Metal and the CPU are compute-bound at this length and run the windows
/// one after another, as before: on an M-series Mac two stacked windows
/// encoded a 600 s file in 17.3 s against 14.7 s one at a time.
fn window_batch_rows(backend: BackendKind) -> usize {
    match backend {
        BackendKind::Cuda if window_batching_enabled() => 4,
        _ => 1,
    }
}

/// The `(rows, frames)` one batch of a long input's `windows` windows of
/// `frames` frames runs at on `backend`.
fn window_batch_shape(windows: usize, frames: usize, backend: BackendKind) -> (usize, usize) {
    encoder_batch_shape(
        windows.clamp(1, window_batch_rows(backend)),
        frames,
        backend,
    )
}

/// Peak preparation memory beyond the weights and the retained artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParakeetPreparationBytes {
    /// Host bytes: the log-mel pipeline of a whole request (mono copy,
    /// pre-emphasis and padded copies, power spectrum and log-mel) in
    /// [`parakeet_preparation_bytes`]; one window's normalized feature copy
    /// in a device job's [`parakeet_encode_row_bytes`].
    pub(crate) host_bytes: u64,
    /// One encoder window's peak: the first subsampling convolution and its
    /// ReLU copy over one stem slice, the attention scores and `rel_shift`
    /// buffers, and the per-frame activations of a layer.
    pub(crate) window_bytes: u64,
    /// The stitched `[1, E, D]` output, copied into the exact-size retained
    /// artifact at the end.
    pub(crate) output_bytes: u64,
}

impl ParakeetPreparationBytes {
    pub(crate) fn device_bytes(self) -> Result<u64> {
        self.window_bytes
            .checked_add(self.output_bytes)
            .ok_or_else(|| Error::Overloaded("Parakeet preparation bytes overflowed".into()))
    }
}

/// Prices the windowed encode of `samples` 16 kHz samples. Windows bound
/// the quadratic attention terms at `W` frames whatever the length; Metal
/// pools round every buffer up to a power of two.
pub(crate) fn parakeet_preparation_bytes(
    samples: usize,
    backend: BackendKind,
) -> Result<ParakeetPreparationBytes> {
    const HOST_BYTES_PER_SAMPLE: u64 = 22;
    let valid_frames = preprocessor::valid_feature_frames(samples);
    // The STFT emits one padding frame past the valid ones.
    let window = subsampled_len_3x(valid_frames + 1).min(EncoderWindowing::V3.frames);
    let host_bytes = u64::try_from(samples)
        .ok()
        .and_then(|samples| samples.checked_mul(HOST_BYTES_PER_SAMPLE))
        .ok_or_else(|| Error::Overloaded("Parakeet preparation bytes overflowed".into()))?;
    // A lone window of one 82 s window or less runs at its graph's shape; a
    // longer input stacks its windows as the rows of one batch.
    let windows = encoder_window_count(subsampled_len_3x(valid_frames), EncoderWindowing::V3)?;
    let (batch_rows, window) = window_batch_shape(windows, window, backend);
    Ok(ParakeetPreparationBytes {
        host_bytes,
        window_bytes: encoder_window_row_bytes(window, batch_rows, 1, backend)?,
        output_bytes: encoder_output_bytes(subsampled_len_3x(valid_frames), backend)?,
    })
}

/// The `(rows, frames)` the encoder runs a batch of `rows` rows padded to
/// `padded_frames` at: single-window rows at their graph's shape where
/// [`encoder_batch_shape`] gives one; a longer row (always alone) stacks
/// its windows as the rows of one batch ([`window_batch_shape`]).
fn encode_row_shape(
    features: &ParakeetHostFeatures,
    padded_frames: usize,
    rows: usize,
    backend: BackendKind,
) -> (usize, usize) {
    if features.single_window() {
        encoder_batch_shape(rows, padded_frames, backend)
    } else {
        window_batch_shape(features.encoder_windows(), padded_frames, backend)
    }
}

/// Prices one row's device job in an encoder batch of `rows` rows whose
/// windows are padded to `padded_frames` encoder frames (on CUDA, to the
/// graph shape [`encoder_batch_shape`] gives): its share of the
/// batch's window peak, its own retained output, and on the host the
/// normalized feature copy each window uploads. The log-mel pipeline itself
/// ran on a host worker before the row was admitted.
pub(crate) fn parakeet_encode_row_bytes(
    features: &ParakeetHostFeatures,
    padded_frames: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<ParakeetPreparationBytes> {
    const F32_BYTES: usize = 4;
    let columns = features
        .feature_frames()
        .min(features.window_frames().saturating_mul(SUBSAMPLING_FACTOR));
    // A long row uploads the features of every window it stacks.
    let uploads = if features.single_window() {
        1
    } else {
        features
            .encoder_windows()
            .clamp(1, window_batch_rows(backend))
    };
    let host_bytes = EncoderGeometry::V3
        .mel_bins
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(uploads))
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("Parakeet preparation bytes overflowed".into()))?;
    let (batch_rows, frames) = encode_row_shape(features, padded_frames, rows, backend);
    Ok(ParakeetPreparationBytes {
        host_bytes,
        window_bytes: encoder_window_row_bytes(frames, batch_rows, rows, backend)?,
        output_bytes: encoder_output_bytes(features.encoded_len(), backend)?,
    })
}

/// Useful and materialized encoder-output elements of one row in a batch
/// of `rows` rows padded to `padded_frames`: a single-window row is padded
/// up to the batch's longest row (on CUDA, to its graph's frames), a longer
/// row (always alone) never.
pub(crate) fn parakeet_encode_row_elements(
    features: &ParakeetHostFeatures,
    padded_frames: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<(u64, u64)> {
    let (_, padded_frames) = encode_row_shape(features, padded_frames, rows, backend);
    let padding = padded_frames.saturating_sub(features.window_frames());
    let elements = |frames: usize| {
        frames
            .checked_mul(EncoderGeometry::V3.dim)
            .and_then(|elements| u64::try_from(elements).ok())
            .ok_or_else(|| Error::Overloaded("Parakeet preparation geometry overflowed".into()))
    };
    let encoded_len = features.encoded_len();
    Ok((
        elements(encoded_len)?,
        elements(encoded_len.saturating_add(padding))?,
    ))
}

/// One row's share of the peak of an encoder batch of `batch_rows` windows
/// of `window` frames that holds `rows` real rows (the rest pad it to a
/// graph's width).
fn encoder_window_row_bytes(
    window: usize,
    batch_rows: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<u64> {
    let bytes = encoder_window_batch_bytes(window, batch_rows, backend)?;
    let rows = u64::try_from(rows.max(1))
        .map_err(|_| Error::Overloaded("Parakeet preparation bytes overflowed".into()))?;
    Ok(bytes
        .stem
        .checked_add(bytes.layers)
        .ok_or_else(|| Error::Overloaded("Parakeet preparation bytes overflowed".into()))?
        .div_ceil(rows))
}

/// The peak of an encoder batch of `rows` windows of `window` frames: the
/// stem runs row by row, so its planes count once per batch; the attention
/// and layer buffers hold every row. Metal rounds each whole batch buffer up
/// to a power of two. Eager attention's score and `rel_shift` planes count
/// unless the fused CUDA kernel passed its self-check
/// ([`relpos_attention::fused_priced`]); the encoder runs that kernel then.
fn encoder_window_batch_bytes(
    window: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<EncoderBatchBytes> {
    let fused = relpos_attention::fused_priced(backend);
    encoder_window_batch_bytes_with(window, rows, backend, fused)
}

/// [`encoder_window_batch_bytes`] with the fused attention on or off.
fn encoder_window_batch_bytes_with(
    window: usize,
    rows: usize,
    backend: BackendKind,
    fused_attention: bool,
) -> Result<EncoderBatchBytes> {
    const F32_BYTES: u64 = 4;
    // About six [T, ff_dim] activations are live inside a layer at once: the
    // feed-forward block's linear, sigmoid and product, plus residual and
    // convolution-module buffers (report 11 sizes them as 98,304·T bytes).
    const LAYER_ACTIVATIONS: u64 = 6;
    let geometry = EncoderGeometry::V3;
    let overflow = || Error::Overloaded("Parakeet preparation bytes overflowed".into());
    let round = |bytes: u64| -> Result<u64> {
        if backend == BackendKind::Metal {
            bytes.checked_next_power_of_two().ok_or_else(overflow)
        } else {
            Ok(bytes)
        }
    };
    let as_u64 = |value: usize| u64::try_from(value).map_err(|_| overflow());
    let (window, rows) = (as_u64(window)?, as_u64(rows.max(1))?);
    let (channels, mel_bins) = (
        as_u64(geometry.subsampling_channels)?,
        as_u64(geometry.mel_bins)?,
    );
    let (heads, dim, ff_dim) = (
        as_u64(geometry.heads)?,
        as_u64(geometry.dim)?,
        as_u64(geometry.ff_dim)?,
    );
    let multiply = |values: &[u64]| {
        values
            .iter()
            .try_fold(1u64, |product, value| product.checked_mul(*value))
            .ok_or_else(overflow)
    };
    // conv0's output [channels, 8T/2, mels/2], its bias add and its ReLU
    // (all three were held on Metal), over one stem slice
    // (`ConvSubsamplingDw::forward`) of T frames and its halos.
    let stem_frames = window.min(as_u64(
        SUBSAMPLING_SLICE_FRAMES + 2 * SUBSAMPLING_HALO_FRAMES,
    )?);
    let stem_plane = multiply(&[channels, 4 * stem_frames, mel_bins / 2, F32_BYTES])?;
    let activation = multiply(&[rows, window, ff_dim, F32_BYTES])?;
    let mut layers = vec![multiply(&[LAYER_ACTIVATIONS, round(activation)?])?];
    if fused_attention {
        // The fused kernel keeps no score plane; the positional projection
        // p is read in place, once for every row.
        layers.push(round(multiply(&[dim, 2 * window, F32_BYTES])?)?);
    } else {
        layers.push(round(multiply(&[rows, heads, window, window, F32_BYTES])?)?);
        // q_v·pᵀ, its zero pad and the reshaped copy, each [H, T, 2T].
        let rel_shift = multiply(&[rows, heads, window, 2 * window, F32_BYTES])?;
        layers.push(multiply(&[3, round(rel_shift)?])?);
        if rows > 1 {
            // The positional projection pᵀ, copied out for every row.
            layers.push(round(multiply(&[rows, dim, 2 * window, F32_BYTES])?)?);
        }
    }
    let layers = layers
        .into_iter()
        .try_fold(0u64, |total, bytes| total.checked_add(bytes))
        .ok_or_else(overflow)?;
    Ok(EncoderBatchBytes {
        stem: multiply(&[3, round(stem_plane)?])?,
        layers,
    })
}

/// The peak of an encoder batch, in its two parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct EncoderBatchBytes {
    /// The stem's planes over one slice of one row.
    stem: u64,
    /// The conformer layers' attention and activation buffers over every
    /// row: a graphed batch's working set.
    layers: u64,
}

/// Kill switch for the Parakeet encoder's CUDA graphs (a boolean; on when
/// unset). Read once: admission prices the graphed shapes from it, and the
/// encoder captures them.
pub(crate) const PARAKEET_ENCODER_GRAPHS_ENV: &str = "IZWI_PARAKEET_ENCODER_GRAPHS";

fn encoder_graphs_requested() -> bool {
    static ON: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or(PARAKEET_ENCODER_GRAPHS_ENV, true));
    ON.get()
}

/// Batch widths the encoder graphs are captured at. A graphed batch pads
/// its rows up to the next, so a busy server's varying widths share ten
/// graphs per length bucket rather than one graph per width.
const ENCODER_GRAPH_ROWS: [usize; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

const MIB: u64 = 1024 * 1024;

/// What the encoder graphs may hold. The pool budget bounds a graph's
/// working set ([`EncoderBatchBytes::layers`]): 32 rows of 128 frames
/// (10 s clips) need 528 MiB, so every width of 10 s clips, 12 rows of
/// 20 s clips and a lone 82 s window are graphed. The retained budget holds
/// the kept graphs' input copies and outputs: 33 MiB for 32 rows of 128
/// frames.
const ENCODER_GRAPH_LIMITS: ShapeGraphLimits = ShapeGraphLimits {
    max_graphs: 24,
    retained_bytes: 192 * MIB,
    pool_bytes: 576 * MIB,
};

/// Device bytes a load on `backend` prices for the encoder graphs, which
/// allocate lazily on their first batches: zero off CUDA or with the graphs
/// switched off.
pub(crate) fn parakeet_encoder_graph_device_bytes(backend: BackendKind) -> u64 {
    if backend == BackendKind::Cuda && encoder_graphs_requested() {
        ENCODER_GRAPH_LIMITS.device_bytes()
    } else {
        0
    }
}

/// The `(rows, frames)` a graph runs a batch of `rows` single-window rows
/// whose longest has `frames` frames at: rows up to the next graph width,
/// frames up to a power of two. `None` past 32 rows, past one window, or
/// past the pool budget.
fn encoder_graph_shape(rows: usize, frames: usize) -> Option<(usize, usize)> {
    let rows = ENCODER_GRAPH_ROWS
        .into_iter()
        .find(|width| *width >= rows.max(1))?;
    let frames = frames.max(1).checked_next_power_of_two()?;
    if frames > EncoderWindowing::V3.frames {
        return None;
    }
    let bytes = encoder_window_batch_bytes(frames, rows, BackendKind::Cuda).ok()?;
    (bytes.layers <= ENCODER_GRAPH_LIMITS.pool_bytes).then_some((rows, frames))
}

/// The `(rows, frames)` an encoder batch of `rows` single-window rows
/// padded to `frames` frames runs at on `backend`: a graph's shape on CUDA
/// with the encoder graphs on, else as given. Admission prices this shape;
/// the encoder runs it, or the given one once its graphs are off.
fn encoder_batch_shape(rows: usize, frames: usize, backend: BackendKind) -> (usize, usize) {
    if backend == BackendKind::Cuda && encoder_graphs_requested() {
        if let Some(shape) = encoder_graph_shape(rows, frames) {
            return shape;
        }
    }
    (rows, frames)
}

/// The retained `[1, E, D]` encoder output.
fn encoder_output_bytes(encoded_len: usize, backend: BackendKind) -> Result<u64> {
    let bytes = encoded_len
        .checked_mul(EncoderGeometry::V3.dim)
        .and_then(|elements| elements.checked_mul(4))
        .and_then(|bytes| u64::try_from(bytes).ok());
    match (bytes, backend) {
        (Some(bytes), BackendKind::Metal) => bytes.checked_next_power_of_two(),
        (bytes, _) => bytes,
    }
    .ok_or_else(|| Error::Overloaded("Parakeet preparation bytes overflowed".into()))
}

pub struct ParakeetAsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Clone)]
pub(crate) struct ParakeetPreparedEncoderArtifact {
    encoded: Tensor,
    encoded_len: usize,
    input_samples: usize,
    input_sample_rate: u32,
    resampled_samples: usize,
    feature_frames: usize,
    language: Option<String>,
    windowing: EncoderWindowing,
    encoder_windows: usize,
    log_mel_ms: f64,
    encode_ms: f64,
}

impl ParakeetPreparedEncoderArtifact {
    /// The device holding the encoder output.
    pub(crate) fn device_location(&self) -> candle_core::DeviceLocation {
        self.encoded.device().location()
    }

    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        let mut accounting = TensorStorageAccounting::default();
        accounting.add_tensor(&self.encoded).ok_or_else(|| {
            Error::Overloaded("Parakeet encoder artifact storage accounting overflowed".into())
        })?;
        Ok(accounting.bytes())
    }
}

/// One request's encoder input, computed on a host worker: the 16 kHz
/// log-mel of its audio, before per-window normalization.
pub(crate) struct ParakeetHostFeatures {
    log_mel: ParakeetLogMel,
    input_samples: usize,
    input_sample_rate: u32,
    resampled_samples: usize,
    log_mel_ms: f64,
    /// Frames of one encoder window.
    window_frames: usize,
    /// Encoder windows the input spans (one when it fits a window).
    windows: usize,
}

impl ParakeetHostFeatures {
    /// Encoder frames (E) these features encode to.
    pub(crate) fn encoded_len(&self) -> usize {
        subsampled_len_3x(self.log_mel.valid_frames())
    }

    /// Whether the audio fits one encoder window, the only rows a batch of
    /// more than one request holds.
    pub(crate) fn single_window(&self) -> bool {
        self.encoded_len() <= self.window_frames
    }

    /// Encoder frames attention spans at once: the whole input when it fits
    /// one window, else one window.
    pub(crate) fn window_frames(&self) -> usize {
        self.encoded_len().min(self.window_frames)
    }

    /// Encoder windows the input spans: one when it fits a window.
    pub(crate) const fn encoder_windows(&self) -> usize {
        self.windows
    }

    /// The input sample count, at the request's own rate.
    pub(crate) const fn input_samples(&self) -> usize {
        self.input_samples
    }

    /// Log-mel columns, including the STFT padding column.
    pub(crate) const fn feature_frames(&self) -> usize {
        self.log_mel.frames()
    }

    /// Host bytes the features hold until the encode drops them.
    pub(crate) fn host_bytes(&self) -> Result<u64> {
        self.log_mel
            .resident_bytes()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Overloaded("Parakeet log-mel bytes overflowed".into()))
    }

    fn artifact(
        &self,
        encoded: Tensor,
        encoder_windows: usize,
        language: Option<&str>,
        windowing: EncoderWindowing,
        encode_ms: f64,
    ) -> ParakeetPreparedEncoderArtifact {
        ParakeetPreparedEncoderArtifact {
            encoded,
            encoded_len: self.encoded_len(),
            input_samples: self.input_samples,
            input_sample_rate: self.input_sample_rate,
            resampled_samples: self.resampled_samples,
            feature_frames: self.log_mel.valid_frames(),
            language: language.map(ToOwned::to_owned),
            windowing,
            encoder_windows,
            log_mel_ms: self.log_mel_ms,
            encode_ms,
        }
    }
}

/// Encoder frames (E) the front end produces for `samples` at `sample_rate`:
/// the same resampled length, `N / hop` valid feature frames and ×8
/// subsampling the encoder itself uses, so preparation can size the decode
/// before any audio is encoded.
pub(crate) fn parakeet_encoded_len_for_samples(samples: usize, sample_rate: u32) -> usize {
    let resampled = resampled_len(samples, sample_rate, SAMPLE_RATE);
    subsampled_len_3x(preprocessor::valid_feature_frames(resampled))
}

/// Upper bound on the labels TDT greedy decoding can emit: at most
/// `max_symbols` per encoder frame, so `E × max_symbols` bounds every
/// transcript of this input.
pub(crate) fn parakeet_max_emittable_tokens(
    samples: usize,
    sample_rate: u32,
    max_symbols: usize,
) -> Result<usize> {
    if sample_rate == 0 {
        return Err(Error::InvalidInput(
            "Parakeet sample rate must be greater than zero".into(),
        ));
    }
    let encoded_len = parakeet_encoded_len_for_samples(samples, sample_rate);
    if encoded_len == 0 {
        return Err(Error::InvalidInput(
            "Parakeet audio is shorter than one 80 ms encoder frame".into(),
        ));
    }
    encoded_len
        .checked_mul(max_symbols.max(1))
        .ok_or_else(|| Error::Overloaded("Parakeet decode budget overflowed".into()))
}

#[derive(Clone)]
pub(crate) struct ParakeetRetainedDecodeState {
    predictor: ParakeetPredictorBatchState,
    predictor_out: Tensor,
    /// The TDT cursor: the next frame, the last emission's frame and count,
    /// and the joint steps taken.
    cursor: TdtRow,
    token_ids: Vec<usize>,
    token_frames: Vec<TdtTokenFrame>,
    detokenizer: IncrementalDetokenizer,
    assembled: String,
    counters: ParakeetDecodeCounters,
    finished: bool,
    stop_reason: Option<&'static str>,
}

pub(crate) struct ParakeetRetainedDecodeBatchRow<'a> {
    pub(crate) artifact: &'a ParakeetPreparedEncoderArtifact,
    pub(crate) state: &'a mut ParakeetRetainedDecodeState,
    /// The request's decode budget. The scheduler stops scheduling a row at
    /// this many tokens, so the row must finish here rather than strand.
    pub(crate) max_tokens: usize,
    /// Tokens this engine step may emit for the row (the scheduler's
    /// quantum, at least one). The eager step emits at most one.
    pub(crate) quota: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParakeetRetainedDecodeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

#[derive(Debug, Default, Clone)]
struct ParakeetTimings {
    resample_ms: f64,
    feature_extract_ms: f64,
    feature_upload_ms: f64,
    pre_encode_ms: f64,
    pos_emb_ms: f64,
    encoder_layers_ms: f64,
    tdt_decode_ms: f64,
    model_total_ms: f64,
}

impl ParakeetTimings {
    fn mel_prepare_ms(&self) -> f64 {
        self.feature_extract_ms + self.feature_upload_ms
    }

    fn encoder_forward_ms(&self) -> f64 {
        self.pre_encode_ms + self.pos_emb_ms + self.encoder_layers_ms
    }
}

#[derive(Debug, Default, Clone)]
struct ParakeetDecodeCounters {
    tdt_joint_steps: usize,
    tdt_emitted_tokens: usize,
    tdt_blank_steps: usize,
    host_argmax_reads: usize,
    device_argmax_scalar_reads: usize,
    /// Tokens passed to the text decoder: O(emitted tokens) with the
    /// incremental detokenizer.
    detokenized_tokens: u64,
}

pub struct ParakeetAsrModel {
    variant: ModelVariant,
    _artifacts: Option<ParakeetArtifacts>,
    decoder: ParakeetDecoder,
    preprocessor: ParakeetPreprocessor,
    network: ParakeetNetwork,
    blank_idx: usize,
    num_durations: usize,
    max_symbols: usize,
    encoder_windowing: EncoderWindowing,
    weight_precision: WeightOnlyF16Load,
}

enum ParakeetDecoder {
    Vocab(Vec<String>),
    HfTokenizer(Tokenizer),
}

impl ParakeetDecoder {
    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Self::Vocab(vocab) => decode_tokens(ids, vocab),
            Self::HfTokenizer(tokenizer) => {
                let token_ids: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
                tokenizer.decode(&token_ids).unwrap_or_default()
            }
        }
    }

    /// Whether the SentencePiece piece for `id` opens a word (`▁` prefix).
    fn starts_word(&self, id: usize) -> bool {
        match self {
            Self::Vocab(vocab) => vocab.get(id).is_some_and(|piece| piece.starts_with('▁')),
            Self::HfTokenizer(tokenizer) => u32::try_from(id)
                .ok()
                .and_then(|id| tokenizer.id_to_token(id))
                .is_some_and(|piece| piece.starts_with('▁')),
        }
    }

    /// Word timings in the `timestamp_words` shape the API reads, from each
    /// emitted token's encoder frame and TDT duration.
    fn timestamp_words(
        &self,
        token_ids: &[usize],
        token_frames: &[TdtTokenFrame],
    ) -> serde_json::Value {
        let words = tdt_word_timestamps(
            token_ids,
            token_frames,
            |id| self.starts_word(id),
            |ids| self.decode(ids),
        );
        serde_json::Value::Array(
            words
                .into_iter()
                .map(|word| {
                    serde_json::json!({
                        "word": word.text,
                        "start_time_seconds": word.start_frame as f64 * ENCODER_FRAME_SECONDS,
                        "end_time_seconds": word.end_frame as f64 * ENCODER_FRAME_SECONDS,
                    })
                })
                .collect(),
        )
    }
}

/// Where TDT emitted a token: the encoder frame the joint read and the
/// duration (in frames) it predicted with the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TdtTokenFrame {
    frame: usize,
    duration: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TdtWordTimestamp {
    text: String,
    start_frame: usize,
    end_frame: usize,
}

/// Group emitted tokens into words as NeMo does for TDT timestamps: a word
/// starts at its first token's frame and ends at its last token's frame plus
/// that token's duration. Punctuation-only words join the word before them.
fn tdt_word_timestamps(
    token_ids: &[usize],
    token_frames: &[TdtTokenFrame],
    starts_word: impl Fn(usize) -> bool,
    decode: impl Fn(&[usize]) -> String,
) -> Vec<TdtWordTimestamp> {
    let count = token_ids.len().min(token_frames.len());
    let mut words: Vec<TdtWordTimestamp> = Vec::new();
    let mut start = 0usize;
    for end in 1..=count {
        if end < count && !starts_word(token_ids[end]) {
            continue;
        }
        let group = start..end;
        start = end;
        let text = decode(&token_ids[group.clone()]).trim().to_string();
        if text.is_empty() {
            continue;
        }
        let start_frame = token_frames[group.start].frame;
        let end_frame = token_frames[group]
            .iter()
            .map(|emission| emission.frame + emission.duration)
            .max()
            .unwrap_or(start_frame)
            .max(start_frame);
        match words.last_mut() {
            Some(previous) if !text.chars().any(char::is_alphanumeric) => {
                previous.text.push_str(&text);
                previous.end_frame = previous.end_frame.max(end_frame);
            }
            _ => words.push(TdtWordTimestamp {
                text,
                start_frame,
                end_frame,
            }),
        }
    }
    words
}

/// One request of a Parakeet encoder batch.
pub(crate) struct ParakeetEncoderBatchRow<'a> {
    pub(crate) features: &'a ParakeetHostFeatures,
    pub(crate) language: Option<&'a str>,
    /// Fails when the request is cancelled or past its deadline.
    pub(crate) checkpoint: &'a dyn Fn() -> Result<()>,
}

impl ParakeetAsrModel {
    pub(crate) fn new_predictor_batch_state(
        &self,
        batch: usize,
    ) -> Result<ParakeetPredictorBatchState> {
        if batch == 0 {
            return Err(Error::InvalidInput(
                "Parakeet predictor batch must contain at least one row".into(),
            ));
        }
        self.network
            .predictor
            .initial_state(batch, self.network.predictor.embed.device())
    }

    /// Execute one native recurrent predictor step for independent request
    /// rows. This is a real B-wide LSTM call; callers retain scalar fallback
    /// by passing one label and one-row state.
    pub(crate) fn predictor_step_batch(
        &self,
        labels: &[usize],
        state: &mut ParakeetPredictorBatchState,
    ) -> Result<Tensor> {
        self.network
            .predictor
            .step_batch(labels, state, self.network.predictor.embed.device())
    }

    /// Run the joint network across independent rows without padding a time
    /// dimension. `encoded_rows` is `[B,Denc]` and `predictor_rows` is
    /// `[B,1,Dpred]`; the result is `[B,V+durations]`.
    pub(crate) fn joint_batch_rows(
        &self,
        encoded_rows: &Tensor,
        predictor_rows: &Tensor,
    ) -> Result<Tensor> {
        self.network
            .joint
            .joint_batch_rows(encoded_rows, predictor_rows)
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<ParakeetPhysicalStateSpec> {
        physical::parakeet_physical_state_spec(stage_graphs)
    }

    /// Labels TDT greedy decoding may emit per encoder frame.
    pub(crate) const fn max_symbols_per_frame(&self) -> usize {
        self.max_symbols
    }

    /// Encodes the audio into the artifact the retained decode reads.
    /// `checkpoint` runs between encoder windows; its error aborts the encode.
    #[cfg(test)]
    pub(crate) fn prepare_retained_encoder(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetPreparedEncoderArtifact> {
        let features = self.prepare_host_features(audio, sample_rate)?;
        self.encode_host_features(&features, language, checkpoint)
    }

    /// Resamples the audio to 16 kHz and computes its log-mel on the host,
    /// so a request's device job runs only the encoder.
    pub(crate) fn prepare_host_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<ParakeetHostFeatures> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".into()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Parakeet sample rate must be greater than zero".into(),
            ));
        }
        let log_mel_started = Instant::now();
        let mono = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };
        let resampled_samples = mono.len();
        let log_mel = self.preprocessor.compute_log_mel(&mono)?;
        drop(mono);
        let log_mel_ms = elapsed_ms(log_mel_started);
        let encoded_len = subsampled_len_3x(log_mel.valid_frames());
        // Preparation sized the decode budget from the sample count alone.
        let budgeted_len = parakeet_encoded_len_for_samples(audio.len(), sample_rate);
        if encoded_len != budgeted_len {
            return Err(Error::InferenceError(format!(
                "Parakeet encoder produced {encoded_len} frames but its decode budget assumed {budgeted_len}"
            )));
        }
        Ok(ParakeetHostFeatures {
            log_mel,
            input_samples: audio.len(),
            input_sample_rate: sample_rate,
            resampled_samples,
            log_mel_ms,
            window_frames: self.encoder_windowing.frames,
            windows: encoder_window_count(encoded_len, self.encoder_windowing)?,
        })
    }

    /// Encodes one request's host features into the artifact the retained
    /// decode reads, window by window. `checkpoint` runs before every
    /// window; its error aborts the encode.
    pub(crate) fn encode_host_features(
        &self,
        features: &ParakeetHostFeatures,
        language: Option<&str>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetPreparedEncoderArtifact> {
        let encode_started = Instant::now();
        let encoded = self.encode_log_mel(
            &features.log_mel,
            &mut ParakeetTimings::default(),
            checkpoint,
        )?;
        let encode_ms = elapsed_ms(encode_started);
        Ok(features.artifact(
            deep_copy_tensor_storage(&encoded.encoded)?,
            encoded.windows,
            language,
            self.encoder_windowing,
            encode_ms,
        ))
    }

    /// Encodes the rows of one preparation batch. A single row runs
    /// [`Self::encode_host_features`]; wider batches hold only single-window
    /// rows, which encode as one padded batch. A row whose checkpoint fails
    /// before the encode fails alone; an encoder error fails the batch.
    pub(crate) fn prepare_retained_encoder_batch(
        &self,
        rows: &[ParakeetEncoderBatchRow<'_>],
    ) -> Result<Vec<Result<ParakeetPreparedEncoderArtifact>>> {
        if let [row] = rows {
            return match self.encode_host_features(row.features, row.language, row.checkpoint) {
                Ok(artifact) => Ok(vec![Ok(artifact)]),
                Err(error @ (Error::Cancelled(_) | Error::Timeout(_))) => Ok(vec![Err(error)]),
                Err(error) => Err(error),
            };
        }
        if rows.iter().any(|row| !row.features.single_window()) {
            return Err(Error::InvalidInput(
                "Parakeet encoder batches hold only single-window rows".into(),
            ));
        }
        let mut outcomes = rows
            .iter()
            .map(|row| (row.checkpoint)().err().map(Err))
            .collect::<Vec<Option<Result<ParakeetPreparedEncoderArtifact>>>>();
        let live = (0..rows.len())
            .filter(|index| outcomes[*index].is_none())
            .collect::<Vec<_>>();
        if let [index] = live[..] {
            let row = &rows[index];
            outcomes[index] =
                Some(self.encode_host_features(row.features, row.language, row.checkpoint));
        } else if !live.is_empty() {
            let device = self.preprocessor.device();
            let encode_started = Instant::now();
            let inputs = live
                .iter()
                .map(|index| {
                    let features = rows[*index].features;
                    let encoded_len = features.encoded_len();
                    let columns = EncoderWindow::whole(encoded_len)
                        .feature_columns(features.feature_frames());
                    let window = features.log_mel.normalize_and_upload(columns, device)?;
                    Ok((window, encoded_len))
                })
                .collect::<Result<Vec<_>>>()?;
            let encoded = self.network.encoder.encode_single_window_batch(&inputs)?;
            drop(inputs);
            let encode_ms = elapsed_ms(encode_started);
            for (index, encoded) in live.into_iter().zip(encoded) {
                let row = &rows[index];
                outcomes[index] = Some(Ok(row.features.artifact(
                    encoded,
                    1,
                    row.language,
                    self.encoder_windowing,
                    encode_ms,
                )));
            }
        }
        Ok(outcomes
            .into_iter()
            .map(|outcome| outcome.expect("every Parakeet batch row has an outcome"))
            .collect())
    }

    /// Runs the windowed encoder over a host log-mel, normalizing each
    /// window's columns over that window as NVIDIA's chunked inference does.
    fn encode_log_mel(
        &self,
        log_mel: &ParakeetLogMel,
        timings: &mut ParakeetTimings,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetEncoded> {
        let device = self.preprocessor.device();
        self.network.encoder.encode_windows(
            log_mel.frames(),
            subsampled_len_3x(log_mel.valid_frames()),
            self.encoder_windowing,
            window_batch_rows(crate::backends::backend_kind_for_device(device)),
            &mut |columns| log_mel.normalize_and_upload(columns, device),
            timings,
            checkpoint,
        )
    }

    pub(crate) fn start_retained_decode(&self) -> Result<ParakeetRetainedDecodeState> {
        self.start_retained_decode_batch(1)?
            .pop()
            .ok_or_else(|| Error::InferenceError("Parakeet B1 prefill produced no row".into()))
    }

    pub(crate) fn start_retained_decode_batch(
        &self,
        batch: usize,
    ) -> Result<Vec<ParakeetRetainedDecodeState>> {
        let mut cohort = self.new_predictor_batch_state(batch)?;
        let labels = vec![self.blank_idx; batch];
        let outputs = self.predictor_step_batch(&labels, &mut cohort)?;
        let mut predictors = (0..batch)
            .map(|_| self.new_predictor_batch_state(1))
            .collect::<Result<Vec<_>>>()?;
        let mut targets = predictors.iter_mut().collect::<Vec<_>>();
        cohort.scatter_rows(&mut targets)?;
        drop(targets);
        predictors
            .into_iter()
            .enumerate()
            .map(|(index, predictor)| {
                Ok(ParakeetRetainedDecodeState {
                    predictor,
                    predictor_out: outputs.narrow(0, index, 1)?.contiguous()?,
                    cursor: TdtRow::START,
                    token_ids: Vec::new(),
                    token_frames: Vec::new(),
                    detokenizer: IncrementalDetokenizer::new(),
                    assembled: String::new(),
                    counters: ParakeetDecodeCounters::default(),
                    finished: false,
                    stop_reason: None,
                })
            })
            .collect()
    }

    pub(crate) fn hydrate_retained_decode_physical_state(
        &self,
        state: &mut ParakeetRetainedDecodeState,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
    ) -> Result<()> {
        let snapshot = arena
            .read_transaction_base(transaction, physical::PARAKEET_PREDICTOR_STATE_DOMAIN)?
            .ok_or_else(|| {
                Error::InferenceError(
                    "Parakeet retained predictor has no committed physical snapshot".into(),
                )
            })?;
        if snapshot.components.len() != 5
            || snapshot
                .components
                .iter()
                .enumerate()
                .any(|(index, value)| {
                    value.component != StateComponentId::new((index + 1) as u32)
                        || value.tensor.is_none()
                })
        {
            return Err(Error::InferenceError(
                "Parakeet retained predictor snapshot has non-canonical components".into(),
            ));
        }
        let tensor = |index: usize| {
            snapshot.components[index].tensor.clone().ok_or_else(|| {
                Error::InferenceError("Parakeet retained predictor snapshot lost a tensor".into())
            })
        };
        state.predictor = ParakeetPredictorBatchState {
            h0: tensor(0)?,
            c0: tensor(1)?,
            h1: tensor(2)?,
            c1: tensor(3)?,
        };
        state.predictor_out = tensor(4)?;
        Ok(())
    }

    pub(crate) fn stage_retained_decode_physical_state(
        &self,
        state: &ParakeetRetainedDecodeState,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        let expected_cursor = arena
            .read_transaction_base(transaction, physical::PARAKEET_PREDICTOR_STATE_DOMAIN)?
            .map_or(0, |snapshot| snapshot.cursor);
        if target_cursor <= expected_cursor {
            return Err(Error::InferenceError(format!(
                "Parakeet retained predictor cursor did not advance: {expected_cursor} -> {target_cursor}"
            )));
        }
        let tensors = [
            state.predictor.h0.clone(),
            state.predictor.c0.clone(),
            state.predictor.h1.clone(),
            state.predictor.c1.clone(),
            state.predictor_out.clone(),
        ];
        let values = tensors
            .into_iter()
            .enumerate()
            .map(|(index, tensor)| StateComponentValue {
                component: StateComponentId::new((index + 1) as u32),
                tensor: Some(tensor),
            })
            .collect();
        arena.stage_replace(
            transaction,
            physical::PARAKEET_PREDICTOR_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            values,
        )
    }

    pub(crate) fn retained_decode_output(
        &self,
        artifact: &ParakeetPreparedEncoderArtifact,
        state: &ParakeetRetainedDecodeState,
    ) -> ParakeetAsrTranscriptionOutput {
        let mut diagnostics = serde_json::json!({
            "input_samples": artifact.input_samples,
            "input_sample_rate": artifact.input_sample_rate,
            "resampled_samples": artifact.resampled_samples,
            "feature_frames": artifact.feature_frames,
            "encoded_frames": artifact.encoded_len,
            "encoder_windows": artifact.encoder_windows,
            "encoder_window_frames": artifact.windowing.frames,
            "encoder_window_left_context": artifact.windowing.left_context,
            "encoder_window_right_context": artifact.windowing.right_context,
            "log_mel_ms": artifact.log_mel_ms,
            "encode_ms": artifact.encode_ms,
            "tdt_joint_steps": state.counters.tdt_joint_steps,
            "tdt_emitted_tokens": state.counters.tdt_emitted_tokens,
            "detokenized_tokens": state.counters.detokenized_tokens,
            "tdt_blank_steps": state.counters.tdt_blank_steps,
            "host_argmax_reads": state.counters.host_argmax_reads,
            "device_argmax_scalar_reads": state.counters.device_argmax_scalar_reads,
            "weight_precision": self.weight_precision_diagnostics(),
            "encoder_graphs": self.network.encoder.graphs_diagnostics(),
        });
        if let Some(stop_reason) = state.stop_reason {
            diagnostics["stop_reason"] = stop_reason.into();
        }
        // The cursor runs over the whole windowed encoding, so frames are
        // already offsets from the start of the audio.
        diagnostics["timestamp_words"] = self
            .decoder
            .timestamp_words(&state.token_ids, &state.token_frames);
        ParakeetAsrTranscriptionOutput {
            text: state.assembled.clone(),
            language: artifact.language.clone(),
            diagnostics: Some(diagnostics),
        }
    }

    pub(crate) const fn retained_decode_finished(state: &ParakeetRetainedDecodeState) -> bool {
        state.finished
    }

    pub(crate) fn retained_decode_step_batch(
        &self,
        rows: &mut [ParakeetRetainedDecodeBatchRow<'_>],
    ) -> Result<Vec<ParakeetRetainedDecodeStep>> {
        if rows.is_empty() || rows.iter().any(|row| row.state.finished) {
            return Err(Error::InvalidInput(
                "Parakeet retained cohort requires live rows".into(),
            ));
        }
        let checkpoints = rows.iter().map(|row| row.state.clone()).collect::<Vec<_>>();
        if self.device_decode_on() {
            let result = self.retained_decode_step_batch_device(rows);
            if result.is_err() {
                for (row, checkpoint) in rows.iter_mut().zip(checkpoints) {
                    *row.state = checkpoint;
                }
            }
            return result;
        }
        let result = (|| {
            loop {
                let encoded_rows = rows
                    .iter()
                    .map(|row| {
                        if row.state.cursor.t >= row.artifact.encoded_len {
                            return Err(Error::InferenceError(
                                "Parakeet retained cursor exceeded encoder artifact".into(),
                            ));
                        }
                        row.artifact
                            .encoded
                            .i((0, row.state.cursor.t, ..))?
                            .unsqueeze(0)
                            .map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let encoded_refs = encoded_rows.iter().collect::<Vec<_>>();
                let encoded = Tensor::cat(&encoded_refs, 0)?;
                let predictor_rows = rows
                    .iter()
                    .map(|row| &row.state.predictor_out)
                    .collect::<Vec<_>>();
                let predictor = Tensor::cat(&predictor_rows, 0)?;
                let logits = self.joint_batch_rows(&encoded, &predictor)?;
                let (_, output_width) = logits.dims2()?;
                let token_end = self.blank_idx + 1;
                if output_width < token_end + self.num_durations {
                    return Err(Error::InferenceError(
                        "Parakeet retained joint output has invalid geometry".into(),
                    ));
                }
                // One readback for every row's label and duration.
                let selections =
                    tdt_batch_labels_and_durations(&logits, token_end, self.num_durations)?;
                let mut emitted = Vec::new();
                let rules = TdtRules {
                    blank: self.blank_idx,
                    num_durations: self.num_durations,
                    max_symbols: self.max_symbols,
                };
                for (index, row) in rows.iter_mut().enumerate() {
                    let (label, duration) = selections[index];
                    let step = tdt_step(&mut row.state.cursor, label, duration, rules);
                    let counters = &mut row.state.counters;
                    counters.tdt_joint_steps = counters.tdt_joint_steps.saturating_add(1);
                    counters.device_argmax_scalar_reads =
                        counters.device_argmax_scalar_reads.saturating_add(1);
                    match step {
                        TdtStep::Blank => {
                            counters.tdt_blank_steps = counters.tdt_blank_steps.saturating_add(1);
                        }
                        TdtStep::Suppressed => {}
                        TdtStep::Emit {
                            label,
                            frame,
                            duration,
                        } => {
                            row.state.token_ids.push(label);
                            row.state
                                .token_frames
                                .push(TdtTokenFrame { frame, duration });
                            emitted.push((index, label));
                        }
                    }
                }
                if !emitted.is_empty() {
                    let refs = emitted
                        .iter()
                        .map(|(index, _)| &rows[*index].state.predictor)
                        .collect::<Vec<_>>();
                    let mut cohort = ParakeetPredictorBatchState::gather_rows(&refs)?;
                    let labels = emitted.iter().map(|(_, label)| *label).collect::<Vec<_>>();
                    let outputs = self.predictor_step_batch(&labels, &mut cohort)?;
                    let mut scattered = emitted
                        .iter()
                        .map(|(index, _)| rows[*index].state.predictor.clone())
                        .collect::<Vec<_>>();
                    let mut targets = scattered.iter_mut().collect::<Vec<_>>();
                    cohort.scatter_rows(&mut targets)?;
                    drop(targets);
                    for ((cohort_index, (row_index, _)), predictor) in
                        emitted.iter().enumerate().zip(scattered)
                    {
                        rows[*row_index].state.predictor = predictor;
                        rows[*row_index].state.predictor_out =
                            outputs.narrow(0, cohort_index, 1)?.contiguous()?;
                    }
                }
                let steps = self.retained_decode_steps(rows)?;
                // A scheduler quantum produces at most one token per row, but
                // blank-only joint evaluations are consumed inside that
                // quantum. This matches the original greedy decoder and
                // avoids one scheduler transaction per blank encoder frame.
                if !emitted.is_empty() || steps.iter().any(|step| step.finished) {
                    break Ok(steps);
                }
            }
        })();
        if result.is_err() {
            for (row, checkpoint) in rows.iter_mut().zip(checkpoints) {
                *row.state = checkpoint;
            }
        }
        result
    }

    /// Each row's step after its joint steps: the detokenized delta and
    /// text, and whether it finished (its cursor past its frames, every frame
    /// at `max_symbols` labels, its token budget, or the joint-step limit).
    fn retained_decode_steps(
        &self,
        rows: &mut [ParakeetRetainedDecodeBatchRow<'_>],
    ) -> Result<Vec<ParakeetRetainedDecodeStep>> {
        let mut steps = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let mut delta = row
                .state
                .detokenizer
                .step(&row.state.token_ids, |ids| Ok(self.decoder.decode(ids)))?;
            let guard_limit = row
                .artifact
                .encoded_len
                .saturating_mul(self.max_symbols)
                .saturating_mul(8)
                .max(512);
            let tokens = row.state.token_ids.len();
            // Every frame emitting `max_symbols` labels exhausts the
            // input as surely as the cursor passing its end does.
            let exhausted = row.state.cursor.t >= row.artifact.encoded_len
                || tokens >= row.artifact.encoded_len.saturating_mul(self.max_symbols);
            let budget_reached = !exhausted && tokens >= row.max_tokens.max(1);
            if budget_reached {
                row.state.stop_reason = Some("max_tokens");
            }
            row.state.finished =
                exhausted || budget_reached || row.state.cursor.guard_steps >= guard_limit;
            if row.state.finished {
                delta.push_str(
                    &row.state
                        .detokenizer
                        .finish(&row.state.token_ids, |ids| Ok(self.decoder.decode(ids)))?,
                );
            }
            row.state.assembled.push_str(&delta);
            row.state.counters.detokenized_tokens = row.state.detokenizer.decoded_tokens();
            row.state.counters.tdt_emitted_tokens = tokens;
            steps.push(ParakeetRetainedDecodeStep {
                delta,
                text: row.state.assembled.clone(),
                tokens_generated: row.state.token_ids.len(),
                finished: row.state.finished,
            });
        }
        Ok(steps)
    }

    /// The encoder's projection precision for diagnostics: weight-only F16
    /// or F32 as loaded, or the BF16 tier's mixed precision.
    fn weight_precision_diagnostics(&self) -> serde_json::Value {
        let mut diagnostics = self.weight_precision.diagnostics();
        if self.network.encoder.projection_dtype == DType::BF16 {
            diagnostics["projection_weights"] = "BF16".into();
            diagnostics["activations"] =
                "BF16 projection GEMMs; F32 residual stream, norms, depthwise conv and attention core"
                    .into();
        }
        diagnostics
    }

    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device_profile: DeviceProfile,
    ) -> Result<Self> {
        if !variant.is_parakeet() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Parakeet model",
                variant.dir_name()
            )));
        }

        let device = select_device_for_parakeet(&device_profile);
        let (artifacts, decoder, vb) = if variant.is_parakeet_nemo() {
            let artifacts = ensure_parakeet_artifacts(model_dir, variant)?;
            let tokenizer_vocab = decode::load_tokenizer_vocab(&artifacts.tokenizer_vocab_path)?;
            let vb =
                VarBuilder::from_pth(&artifacts.checkpoint_path, PARAKEET_WEIGHT_DTYPE, &device)
                    .map_err(|e| {
                        Error::ModelLoadError(format!(
                            "Failed to load Parakeet checkpoint {}: {}",
                            artifacts.checkpoint_path.display(),
                            e
                        ))
                    })?;
            (Some(artifacts), ParakeetDecoder::Vocab(tokenizer_vocab), vb)
        } else if variant.is_parakeet_mlx() {
            let weights_path = model_dir.join("model.safetensors");
            if !weights_path.exists() {
                return Err(Error::ModelNotFound(format!(
                    "Missing Parakeet MLX weights at {}",
                    weights_path.display()
                )));
            }
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)
            }
            .map_err(|e| {
                Error::ModelLoadError(format!("Failed to load Parakeet MLX weights: {e}"))
            })?;
            (None, load_mlx_decoder(model_dir)?, vb)
        } else {
            return Err(Error::InvalidInput(format!(
                "Unsupported Parakeet variant: {}",
                variant.dir_name()
            )));
        };

        let preprocessor = ParakeetPreprocessor::load(&vb, model_dir)?;
        let projection_precision = resolve_parakeet_projection_precision(&device_profile);
        let (network, weight_precision) = ParakeetNetwork::load(&vb, projection_precision)?;
        tracing::info!(
            "Parakeet encoder projections load in {}: {}",
            network.encoder.projection_weights(),
            weight_precision.reason
        );

        let blank_idx = network.blank_idx;
        let num_durations = network.num_durations;
        // The model loaded from the extracted checkpoint: its `.nemo` is
        // no longer needed.
        if let Some(artifacts) = &artifacts {
            crate::models::shared::weights::nemo::release_extracted_nemo(
                &artifacts.nemo_path,
                nemo::parakeet_native_artifacts_complete(model_dir, variant),
            );
        }

        let model = Self {
            variant,
            _artifacts: artifacts,
            decoder,
            preprocessor,
            network,
            blank_idx,
            num_durations,
            max_symbols: DEFAULT_MAX_SYMBOLS,
            encoder_windowing: EncoderWindowing::V3,
            weight_precision,
        };
        model.check_device_decode();
        Ok(model)
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<ParakeetAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback_internal(audio, sample_rate, language, &mut no_op)
    }

    pub(crate) fn transcribe_with_details_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        predictor_state: &mut InvocationTensorLease,
    ) -> Result<ParakeetAsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback_internal_physical(
            audio,
            sample_rate,
            language,
            predictor_state,
            &mut no_op,
        )
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_with_callback_internal(audio, sample_rate, language, on_delta)?
            .text)
    }

    pub(crate) fn transcribe_with_callback_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        predictor_state: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_with_callback_internal_physical(
                audio,
                sample_rate,
                language,
                predictor_state,
                on_delta,
            )?
            .text)
    }

    fn transcribe_with_callback_internal(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ParakeetAsrTranscriptionOutput> {
        self.transcribe_with_callback_internal_impl(audio, sample_rate, language, None, on_delta)
    }

    fn transcribe_with_callback_internal_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        predictor_state: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ParakeetAsrTranscriptionOutput> {
        if predictor_state.domain() != physical::PARAKEET_PREDICTOR_STATE_DOMAIN {
            return Err(Error::InferenceError(
                "Parakeet ASR received a foreign physical predictor domain".into(),
            ));
        }
        self.transcribe_with_callback_internal_impl(
            audio,
            sample_rate,
            language,
            Some(predictor_state),
            on_delta,
        )
    }

    fn transcribe_with_callback_internal_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        predictor_state: Option<&mut InvocationTensorLease>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<ParakeetAsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let model_started = Instant::now();
        let mut timings = ParakeetTimings::default();
        let resample_started = Instant::now();
        let mono_16khz = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };
        timings.resample_ms = elapsed_ms(resample_started);

        let feature_started = Instant::now();
        let log_mel = self.preprocessor.compute_log_mel(&mono_16khz)?;
        let feature_frames = log_mel.valid_frames();
        timings.feature_extract_ms = elapsed_ms(feature_started);

        let ParakeetEncoded {
            encoded,
            encoded_len,
            windows: encoder_windows,
        } = self.encode_log_mel(&log_mel, &mut timings, &|| Ok(()))?;
        drop(log_mel);

        let mut token_ids = Vec::<usize>::new();
        let mut token_frames = Vec::<TdtTokenFrame>::new();
        let mut assembled = String::new();
        let mut decode_counters = ParakeetDecodeCounters::default();
        let mut detokenizer = IncrementalDetokenizer::new();
        let decode_text = |ids: &[usize]| -> Result<String> { Ok(self.decoder.decode(ids)) };

        let mut on_token = |token_id: usize, emission: TdtTokenFrame| {
            token_ids.push(token_id);
            token_frames.push(emission);
            // The decoder is infallible, so a step cannot fail.
            let delta = detokenizer
                .step(&token_ids, decode_text)
                .unwrap_or_default();
            if !delta.is_empty() {
                on_delta(delta.as_str());
                assembled.push_str(&delta);
            }
        };

        let decode_started = Instant::now();
        match predictor_state {
            Some(state) => self.network.decode_tdt_greedy_physical(
                &encoded,
                encoded_len,
                self.blank_idx,
                self.num_durations,
                self.max_symbols,
                &mut decode_counters,
                state,
                &mut on_token,
            )?,
            None => self.network.decode_tdt_greedy(
                &encoded,
                encoded_len,
                self.blank_idx,
                self.num_durations,
                self.max_symbols,
                &mut decode_counters,
                &mut on_token,
            )?,
        }
        timings.tdt_decode_ms = elapsed_ms(decode_started);

        let held_back = detokenizer.finish(&token_ids, decode_text)?;
        if !held_back.is_empty() {
            on_delta(held_back.as_str());
            assembled.push_str(&held_back);
        }

        decode_counters.detokenized_tokens = detokenizer.decoded_tokens();
        decode_counters.tdt_emitted_tokens = token_ids.len();
        timings.model_total_ms = elapsed_ms(model_started);

        let mut diagnostics = parakeet_diagnostics_json(
            &self.variant,
            self.preprocessor.device(),
            self.network.encoder.geometry,
            self.encoder_windowing,
            encoder_windows,
            &timings,
            &decode_counters,
            sample_rate,
            audio.len(),
            mono_16khz.len(),
            feature_frames,
            encoded_len,
            language,
            self.max_symbols,
        );
        diagnostics["timestamp_words"] = self.decoder.timestamp_words(&token_ids, &token_frames);
        diagnostics["execution"]["weight_precision"] = self.weight_precision_diagnostics();
        diagnostics["execution"]["encoder_graphs"] = self.network.encoder.graphs_diagnostics();

        Ok(ParakeetAsrTranscriptionOutput {
            text: assembled,
            language: language.map(ToString::to_string),
            diagnostics: Some(diagnostics),
        })
    }
}

fn load_mlx_decoder(model_dir: &Path) -> Result<ParakeetDecoder> {
    if let Ok(tokenizer) = Tokenizer::from_path(model_dir) {
        return Ok(ParakeetDecoder::HfTokenizer(tokenizer));
    }

    // Prefer tokenizer.vocab for MLX Parakeet exports. vocab.txt can be a
    // truncated/auxiliary list with different indexing.
    for candidate in ["tokenizer.vocab", "vocab.txt"] {
        let vocab_path = model_dir.join(candidate);
        if vocab_path.exists() {
            let vocab = decode::load_tokenizer_vocab(&vocab_path)?;
            return Ok(ParakeetDecoder::Vocab(vocab));
        }
    }

    Err(Error::ModelLoadError(format!(
        "Unable to locate tokenizer assets for Parakeet MLX model at {}",
        model_dir.display()
    )))
}

fn select_device_for_parakeet(device_profile: &DeviceProfile) -> Device {
    device_profile.device.clone()
}

fn parakeet_diagnostics_json(
    variant: &ModelVariant,
    device: &Device,
    geometry: EncoderGeometry,
    windowing: EncoderWindowing,
    encoder_windows: usize,
    timings: &ParakeetTimings,
    decode: &ParakeetDecodeCounters,
    input_sample_rate: u32,
    input_samples: usize,
    resampled_samples: usize,
    feature_frames: usize,
    encoded_frames: usize,
    requested_language: Option<&str>,
    max_symbols: usize,
) -> serde_json::Value {
    serde_json::json!({
        "model_family": "parakeet_tdt",
        "model": variant.dir_name(),
        "audio": {
            "input_sample_rate": input_sample_rate,
            "sample_rate": SAMPLE_RATE,
            "input_samples": input_samples,
            "resampled_samples": resampled_samples,
            "duration_seconds": resampled_samples as f64 / SAMPLE_RATE as f64,
            "feature_frames": feature_frames,
            "encoded_frames": encoded_frames,
            "subsampling_factor": SUBSAMPLING_FACTOR,
            "frame_hop_ms": FRAME_HOP_MS,
        },
        "prompt": {
            "language": requested_language,
        },
        "decode": {
            "tdt_joint_steps": decode.tdt_joint_steps,
            "tdt_emitted_tokens": decode.tdt_emitted_tokens,
            "tdt_blank_steps": decode.tdt_blank_steps,
            "host_argmax_reads": decode.host_argmax_reads,
            "device_argmax_scalar_reads": decode.device_argmax_scalar_reads,
            "detokenized_tokens": decode.detokenized_tokens,
            "generated_tokens": decode.tdt_emitted_tokens,
            "max_symbols_per_frame": max_symbols,
        },
        "execution": {
            "device_kind": parakeet_device_kind(device),
            "encoder_layers": geometry.layers,
            "encoder_dim": geometry.dim,
            "encoder_heads": geometry.heads,
            "encoder_head_dim": geometry.head_dim(),
            "conformer_conv_kernel": geometry.conv_kernel,
            "depthwise_conv1d_groups": geometry.dim,
            "subsampling_channels": geometry.subsampling_channels,
            "encoder_windows": encoder_windows,
            "encoder_window_frames": windowing.frames,
            "encoder_window_left_context": windowing.left_context,
            "encoder_window_right_context": windowing.right_context,
            "metal_softmax_last_dim": device.is_metal(),
            "metal_depthwise_conv1d": metal_kernels::depthwise_conv1d_enabled_for_device(device),
            "metal_depthwise_conv2d": metal_kernels::depthwise_conv2d_enabled_for_device(device),
            "conformer_conv_fused": metal_kernels::depthwise_conv1d_bias_swish_enabled_for_device(device),
            "shared_depthwise_conv1d_calls_total": crate::kernels::depthwise_conv::depthwise_conv1d_calls(),
            "cuda_depthwise_conv1d_kernel": crate::kernels::depthwise_conv::prepare_cuda_kernel(device),
            "cuda_depthwise_conv1d_kernel_calls_total": crate::kernels::depthwise_conv::depthwise_conv1d_cuda_kernel_calls(),
            "cuda_relpos_attention": relpos_attention::diagnostics(device),
            "cuda_transducer_decode": device_decode::device_decode_diagnostics(device),
        },
        "timings_ms": {
            "resample": timings.resample_ms,
            "feature_extract": timings.feature_extract_ms,
            "feature_upload": timings.feature_upload_ms,
            "mel_prepare": timings.mel_prepare_ms(),
            "mel": timings.mel_prepare_ms(),
            "pre_encode": timings.pre_encode_ms,
            "pos_emb": timings.pos_emb_ms,
            "encoder_layers": timings.encoder_layers_ms,
            "encoder_forward": timings.encoder_forward_ms(),
            "tdt_decode": timings.tdt_decode_ms,
            "decode": timings.tdt_decode_ms,
            "model_total": timings.model_total_ms,
        },
    })
}

fn parakeet_device_kind(device: &Device) -> &'static str {
    if device.is_metal() {
        "Metal"
    } else if device.is_cuda() {
        "Cuda"
    } else {
        "Cpu"
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

struct ParakeetNetwork {
    encoder: ParakeetEncoder,
    predictor: Predictor,
    joint: Joint,
    blank_idx: usize,
    num_durations: usize,
}

impl ParakeetNetwork {
    fn load(
        vb: &VarBuilder,
        projection_precision: DTypeSelection,
    ) -> Result<(Self, WeightOnlyF16Load)> {
        let (encoder, weight_precision) = ParakeetEncoder::load_with_precision(
            vb.pp("encoder"),
            EncoderGeometry::V3,
            projection_precision,
        )?;
        let predictor = Predictor::load(vb.pp("decoder.prediction"))?;
        let joint = Joint::load(
            vb.pp("joint"),
            encoder.geometry.dim,
            predictor.blank_idx + 1,
        )?;

        let blank_idx = predictor.blank_idx;
        let num_durations = joint.num_durations;

        Ok((
            Self {
                encoder,
                predictor,
                joint,
                blank_idx,
                num_durations,
            },
            weight_precision,
        ))
    }

    fn decode_tdt_greedy(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        blank_idx: usize,
        num_durations: usize,
        max_symbols: usize,
        counters: &mut ParakeetDecodeCounters,
        on_token: &mut dyn FnMut(usize, TdtTokenFrame),
    ) -> Result<()> {
        let encoded = encoded.i((0, ..encoded_len, ..))?; // [T, D]

        let mut predictor_state = self.predictor.initial_state(1, encoded.device())?;
        let predictor_out =
            self.predictor
                .step(blank_idx, &mut predictor_state, encoded.device())?;
        self.decode_tdt_greedy_loop(
            &encoded,
            encoded_len,
            blank_idx,
            num_durations,
            max_symbols,
            counters,
            predictor_out,
            |label| {
                self.predictor
                    .step(label, &mut predictor_state, encoded.device())
            },
            on_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tdt_greedy_physical(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        blank_idx: usize,
        num_durations: usize,
        max_symbols: usize,
        counters: &mut ParakeetDecodeCounters,
        predictor_state: &mut InvocationTensorLease,
        on_token: &mut dyn FnMut(usize, TdtTokenFrame),
    ) -> Result<()> {
        let encoded = encoded.i((0, ..encoded_len, ..))?;
        let predictor_out =
            self.predictor
                .initialize_physical(blank_idx, predictor_state, encoded.device())?;
        self.decode_tdt_greedy_loop(
            &encoded,
            encoded_len,
            blank_idx,
            num_durations,
            max_symbols,
            counters,
            predictor_out,
            |label| {
                self.predictor
                    .step_physical(label, predictor_state, encoded.device())
            },
            on_token,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_tdt_greedy_loop(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        blank_idx: usize,
        num_durations: usize,
        max_symbols: usize,
        counters: &mut ParakeetDecodeCounters,
        mut predictor_out: Tensor,
        mut predictor_step: impl FnMut(usize) -> Result<Tensor>,
        on_token: &mut dyn FnMut(usize, TdtTokenFrame),
    ) -> Result<()> {
        let mut t = 0usize;
        let mut last_emit_t = usize::MAX;
        let mut emit_count_at_t = 0usize;
        let mut guard_steps = 0usize;
        let guard_limit = encoded_len
            .saturating_mul(max_symbols)
            .saturating_mul(8)
            .max(512);

        while t < encoded_len && guard_steps < guard_limit {
            guard_steps += 1;

            let t_cur = t;
            // Frame the joint read when it produced `label`.
            let mut frame = t_cur;
            let enc_t = encoded.i((t_cur, ..))?.unsqueeze(0)?.unsqueeze(0)?; // [1,1,D]
            let logits = self
                .joint
                .joint_after_projection(&enc_t, &predictor_out)?
                .squeeze(0)?
                .squeeze(0)?
                .squeeze(0)?; // [V + 1 + num_durations]

            counters.tdt_joint_steps = counters.tdt_joint_steps.saturating_add(1);
            counters.device_argmax_scalar_reads =
                counters.device_argmax_scalar_reads.saturating_add(1);
            let (mut label, duration_idx) = tdt_label_and_duration(&logits, blank_idx + 1)?;
            let mut jump = duration_idx.min(num_durations.saturating_sub(1));

            if label == blank_idx && jump == 0 {
                jump = 1;
            }

            t = t.saturating_add(jump);

            while label == blank_idx && t < encoded_len {
                counters.tdt_blank_steps = counters.tdt_blank_steps.saturating_add(1);
                frame = t;
                let enc_t = encoded.i((t, ..))?.unsqueeze(0)?.unsqueeze(0)?;
                let logits = self
                    .joint
                    .joint_after_projection(&enc_t, &predictor_out)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .squeeze(0)?;

                counters.tdt_joint_steps = counters.tdt_joint_steps.saturating_add(1);
                counters.device_argmax_scalar_reads =
                    counters.device_argmax_scalar_reads.saturating_add(1);
                let duration_idx;
                (label, duration_idx) = tdt_label_and_duration(&logits, blank_idx + 1)?;
                jump = duration_idx.min(num_durations.saturating_sub(1));

                if label == blank_idx && jump == 0 {
                    jump = 1;
                }

                t = t.saturating_add(jump);
            }

            if label == blank_idx {
                continue;
            }

            if t_cur == last_emit_t {
                emit_count_at_t = emit_count_at_t.saturating_add(1);
            } else {
                last_emit_t = t_cur;
                emit_count_at_t = 1;
            }

            if emit_count_at_t > max_symbols {
                t = t_cur.saturating_add(1);
                continue;
            }

            on_token(
                label,
                TdtTokenFrame {
                    frame,
                    duration: jump,
                },
            );
            predictor_out = predictor_step(label)?;
        }

        Ok(())
    }
}

struct ParakeetEncoder {
    geometry: EncoderGeometry,
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    /// The dtype the projection GEMMs run in: BF16 in the CUDA tier, else
    /// F32 (weight-only F16 projections take and return F32).
    projection_dtype: DType,
    /// CUDA graphs of the conformer layers over a padded single-window
    /// batch, one per `(rows, frames)` shape ([`encoder_graph_shape`]), or
    /// why there are none.
    graphs: std::result::Result<ShapeGraphs<(usize, usize)>, String>,
}

impl ParakeetEncoder {
    /// Loads the encoder with its projections in `projections`' tier.
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(vb.pp("pre_encode"), geometry, projections)?;
        let layers = (0..geometry.layers)
            .map(|idx| ConformerLayer::load(vb.pp(format!("layers.{idx}")), geometry, projections))
            .collect::<Result<Vec<_>>>()?;
        let graphs = if encoder_graphs_requested() {
            ShapeGraphs::resolve("Parakeet encoder", vb.device(), ENCODER_GRAPH_LIMITS)
        } else {
            Err(format!("{PARAKEET_ENCODER_GRAPHS_ENV} is off"))
        };
        Ok(Self {
            geometry,
            pre_encode,
            layers,
            projection_dtype: projections.dtype(),
            graphs,
        })
    }

    /// Whether single-window batches run through the CUDA graphs: on CUDA
    /// with the graphs switched on and no graph failure yet.
    fn graphs_on(&self) -> bool {
        self.graphs.as_ref().is_ok_and(ShapeGraphs::is_on)
    }

    /// The `(rows, frames)` a batch of `rows` single-window rows whose
    /// longest has `frames` frames runs at: its graph's shape while the
    /// graphs are on (at most what admission priced, [`encoder_batch_shape`]),
    /// else as given.
    fn batch_shape(&self, rows: usize, frames: usize) -> (usize, usize) {
        if self.graphs_on() {
            encoder_graph_shape(rows, frames).unwrap_or((rows, frames))
        } else {
            (rows, frames)
        }
    }

    fn graphs_diagnostics(&self) -> serde_json::Value {
        match &self.graphs {
            Ok(graphs) => graphs.diagnostics(),
            Err(reason) => serde_json::json!({ "off": reason }),
        }
    }

    /// Loads the encoder at `precision`: BF16 projections and GEMMs, or
    /// weight-only F16 projections kept only if
    /// [`Self::weight_only_f16_self_check`] passes, or F32.
    fn load_with_precision(
        vb: VarBuilder,
        geometry: EncoderGeometry,
        precision: DTypeSelection,
    ) -> Result<(Self, WeightOnlyF16Load)> {
        let device = vb.device().clone();
        if precision.dtype == DType::BF16 {
            let encoder = Self::load(vb, geometry, Projections::Bf16(&device))?;
            return Ok((encoder, WeightOnlyF16Load::dense(precision.reason)));
        }
        load_weight_only_f16(
            "Parakeet",
            &precision,
            &device,
            |f16| {
                let projections = f16.map_or(Projections::F32, Projections::F16);
                Self::load(vb.clone(), geometry, projections)
            },
            |encoder| encoder.weight_only_f16_self_check(&vb),
        )
    }

    /// The projection weights' dtype, as diagnostics report it.
    fn projection_weights(&self) -> &'static str {
        if self.pre_encode.out.as_f16().is_some() {
            "F16"
        } else if self.projection_dtype == DType::BF16 {
            "BF16"
        } else {
            "F32"
        }
    }

    /// The relative positional embedding of a `frames`-frame input, in the
    /// projection dtype: `linear_pos` reads it in every layer, so it is cast
    /// once per encode, not once per layer.
    fn positions(&self, frames: usize, device: &Device) -> Result<Tensor> {
        build_rel_positional_embedding(frames, self.geometry.dim, device)?
            .to_dtype(self.projection_dtype)
            .map_err(Error::from)
    }

    /// Checks the F16 conversion against the original F32 weight of the first
    /// FFN projection, then runs the encoder on xorshift log-mel features and
    /// requires the subsampling output and every layer output to be finite and
    /// inside the F16-safe range.
    fn weight_only_f16_self_check(&self, vb: &VarBuilder) -> NumericSelfCheckReport {
        run_numeric_self_check("Parakeet weight-only F16", F16_SAFE_MAX_ABS, || {
            let geometry = self.geometry;
            let layer0 = self
                .layers
                .first()
                .ok_or_else(|| Error::ModelLoadError("Parakeet encoder has no layers".into()))?;
            if let Some(converted) = layer0.ff1.linear1.as_f16() {
                let original = mlx::load_linear_no_bias(
                    geometry.dim,
                    geometry.ff_dim,
                    Projections::F16(&Device::Cpu)
                        .vb(vb)
                        .pp("layers.0.feed_forward1.linear1"),
                )?;
                let relative = converted.check_against(original.weight(), None, 8)?;
                if !relative.is_finite() || relative > F16_PROJECTION_MAX_RELATIVE_L2 {
                    return Err(Error::ModelLoadError(format!(
                        "layers.0.feed_forward1.linear1 in F16 differs from its F32 weights by relative L2 {relative:.3e}"
                    )));
                }
            }
            let device = vb.device();
            let features = xorshift_tensor(&[1, geometry.mel_bins, 64], 0xA5A5, 2.0, device)?;
            let mut x = self.pre_encode.forward(&features)?;
            let mut outputs = vec![("pre_encode".to_string(), x.clone())];
            let pos = self.positions(x.dim(1)?, device)?;
            for (index, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x, &pos, None)?;
                outputs.push((format!("layers.{index}"), x.clone()));
            }
            Ok::<_, Error>(outputs)
        })
    }

    /// Encodes `encoded_len` frames window by window (see
    /// [`EncoderWindowing`]). `window_features` returns the
    /// `[1, mel_bins, columns]` features of a log-mel column range, and
    /// `checkpoint` runs before every window so a cancelled or expired
    /// request stops within one window.
    fn encode_windows(
        &self,
        feature_frames: usize,
        encoded_len: usize,
        windowing: EncoderWindowing,
        window_rows: usize,
        window_features: &mut dyn FnMut(Range<usize>) -> Result<Tensor>,
        timings: &mut ParakeetTimings,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetEncoded> {
        let plan = plan_encoder_windows(
            encoded_len,
            windowing.frames,
            windowing.left_context,
            windowing.right_context,
        )?;
        if plan.is_empty() {
            return Err(Error::InferenceError(
                "Parakeet encoder produced no decodable frames".into(),
            ));
        }
        if plan.len() > 1 && window_rows > 1 {
            return self.encode_stacked_windows(
                &plan,
                feature_frames,
                encoded_len,
                window_rows,
                window_features,
                timings,
                checkpoint,
            );
        }
        let mut stitched: Option<Tensor> = None;
        let mut pos_emb: Option<Tensor> = None;
        for window in &plan {
            checkpoint()?;
            let upload_started = Instant::now();
            let features = window_features(window.feature_columns(feature_frames))?;
            timings.feature_upload_ms += elapsed_ms(upload_started);

            let frames = window.end - window.start;
            let pre_encode_started = Instant::now();
            let mut x = self.stem_window(&features, frames)?;
            drop(features);
            timings.pre_encode_ms += elapsed_ms(pre_encode_started);

            if plan.len() == 1 && self.graphs_on() {
                // A lone window runs as a padded batch of one, through its
                // graph; its real frames come back as a view.
                let layers_started = Instant::now();
                let (_, padded) = self.batch_shape(1, frames);
                let x = x.pad_with_zeros(1, 0, padded - frames)?;
                let encoded = self.encode_padded(x, &[frames])?.narrow(1, 0, frames)?;
                timings.encoder_layers_ms += elapsed_ms(layers_started);
                return Ok(ParakeetEncoded {
                    encoded,
                    encoded_len,
                    windows: 1,
                });
            }

            // Relative positions depend only on i - j, so one embedding of
            // the window length serves every window.
            let pos_started = Instant::now();
            let pos = match pos_emb.take() {
                Some(pos) if pos.dim(1)? == 2 * frames - 1 => pos,
                _ => self.positions(frames, x.device())?,
            };
            timings.pos_emb_ms += elapsed_ms(pos_started);

            let layers_started = Instant::now();
            for layer in &self.layers {
                x = layer.forward(&x, &pos, None)?;
            }
            timings.encoder_layers_ms += elapsed_ms(layers_started);
            pos_emb = Some(pos);

            if plan.len() == 1 {
                return Ok(ParakeetEncoded {
                    encoded: x,
                    encoded_len,
                    windows: 1,
                });
            }
            let output = match stitched.take() {
                Some(output) => output,
                None => Tensor::zeros((1, encoded_len, self.geometry.dim), x.dtype(), x.device())?,
            };
            // Copy the kept frames: a narrowed view would keep the whole
            // window's buffer alive until the end of the encode.
            let kept = x.narrow(
                1,
                window.keep_start - window.start,
                window.keep_end - window.keep_start,
            )?;
            output.slice_set(&kept, 1, window.keep_start)?;
            drop((kept, x));
            // Metal returns pooled buffers only at a synchronization point.
            output.device().synchronize()?;
            stitched = Some(output);
        }
        Ok(ParakeetEncoded {
            encoded: stitched.ok_or_else(|| {
                Error::InferenceError("Parakeet windowed encode stitched no frames".into())
            })?,
            encoded_len,
            windows: plan.len(),
        })
    }

    /// [`Self::encode_windows`] over a long input's windows, `rows` at a time
    /// as the rows of one batch ([`Self::encode_single_window_batch`]): every
    /// window has the same frames, so no row is padded, and each window's
    /// kept frames are stitched as the sequential loop stitches them. The
    /// stems run inside the batch, so their time counts as layer time.
    /// `checkpoint` runs before every batch.
    #[allow(clippy::too_many_arguments)]
    fn encode_stacked_windows(
        &self,
        plan: &[EncoderWindow],
        feature_frames: usize,
        encoded_len: usize,
        rows: usize,
        window_features: &mut dyn FnMut(Range<usize>) -> Result<Tensor>,
        timings: &mut ParakeetTimings,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetEncoded> {
        let mut stitched: Option<Tensor> = None;
        for group in plan.chunks(rows.max(1)) {
            checkpoint()?;
            let upload_started = Instant::now();
            let inputs = group
                .iter()
                .map(|window| {
                    Ok((
                        window_features(window.feature_columns(feature_frames))?,
                        window.end - window.start,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            timings.feature_upload_ms += elapsed_ms(upload_started);

            let layers_started = Instant::now();
            let encoded = self.encode_single_window_batch(&inputs)?;
            drop(inputs);
            timings.encoder_layers_ms += elapsed_ms(layers_started);

            let output = match stitched.take() {
                Some(output) => output,
                None => {
                    let first = encoded.first().ok_or_else(|| {
                        Error::InferenceError("Parakeet window batch encoded no rows".into())
                    })?;
                    Tensor::zeros(
                        (1, encoded_len, self.geometry.dim),
                        first.dtype(),
                        first.device(),
                    )?
                }
            };
            for (window, x) in group.iter().zip(encoded) {
                let kept = x.narrow(
                    1,
                    window.keep_start - window.start,
                    window.keep_end - window.keep_start,
                )?;
                output.slice_set(&kept, 1, window.keep_start)?;
            }
            // Metal returns pooled buffers only at a synchronization point.
            output.device().synchronize()?;
            stitched = Some(output);
        }
        Ok(ParakeetEncoded {
            encoded: stitched.ok_or_else(|| {
                Error::InferenceError("Parakeet windowed encode stitched no frames".into())
            })?,
            encoded_len,
            windows: plan.len(),
        })
    }

    /// The stem over one window's `[1, mel_bins, columns]` features, cut to
    /// the window's `frames` encoder frames.
    fn stem_window(&self, features: &Tensor, frames: usize) -> Result<Tensor> {
        let x = self.pre_encode.forward(features)?;
        if x.dim(1)? < frames {
            return Err(Error::InferenceError(format!(
                "Parakeet window produced {} encoder frames, expected {frames}",
                x.dim(1)?
            )));
        }
        // Attention must see only this window's frames, never one made
        // purely of STFT padding.
        x.narrow(1, 0, frames).map_err(Error::from)
    }

    /// Encodes single-window rows, each `(features, frames)` as
    /// [`Self::encode_windows`] takes its one window, in one batch. Each row
    /// runs the stem alone (about 1.5% of the work): padded log-mel columns
    /// would leave its first convolution as ReLU(bias), not zero, and the
    /// next convolution would read them into real frames. The stem outputs
    /// are zero-padded to the longest row and the layers run once over
    /// `[B, T, D]` with the padding masked (see [`EncoderPadding`]). While
    /// the CUDA graphs are on, the batch pads further, to its graph's shape
    /// ([`Self::batch_shape`]): frames to a power of two and zero rows up to
    /// the next graph width, all of whose frames count as real. Each real
    /// row's frames are copied out, so no output keeps the batch buffer
    /// alive.
    fn encode_single_window_batch(&self, rows: &[(Tensor, usize)]) -> Result<Vec<Tensor>> {
        let longest = rows.iter().map(|(_, frames)| *frames).max().unwrap_or(0);
        self.encode_single_window_batch_at(rows, self.batch_shape(rows.len(), longest))
    }

    /// [`Self::encode_single_window_batch`] padded to `(batch_rows, padded)`.
    fn encode_single_window_batch_at(
        &self,
        rows: &[(Tensor, usize)],
        (batch_rows, padded): (usize, usize),
    ) -> Result<Vec<Tensor>> {
        let mut lengths = rows.iter().map(|(_, frames)| *frames).collect::<Vec<_>>();
        if batch_rows < rows.len() || lengths.iter().any(|frames| *frames > padded) {
            return Err(Error::InvalidInput(format!(
                "Parakeet batch rows {lengths:?} do not fit {batch_rows} rows of {padded} frames"
            )));
        }
        let stems = rows
            .iter()
            .map(|(features, frames)| {
                self.stem_window(features, *frames)?
                    .pad_with_zeros(1, 0, padded - frames)
                    .map_err(Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut x = Tensor::cat(&stems, 0)?;
        drop(stems);
        if batch_rows > rows.len() {
            x = x.pad_with_zeros(0, 0, batch_rows - rows.len())?;
            lengths.resize(batch_rows, padded);
        }
        let encoded = self.encode_padded(x, &lengths)?;
        lengths
            .iter()
            .take(rows.len())
            .enumerate()
            .map(|(row, frames)| {
                let frames = encoded.narrow(0, row, 1)?.narrow(1, 0, *frames)?;
                deep_copy_tensor_storage(&frames).map_err(Error::from)
            })
            .collect()
    }

    /// The conformer layers over a zero-padded `[B, T, D]` batch whose row
    /// `b` holds `lengths[b]` real frames: through the CUDA graph of a
    /// graph's shape, eagerly otherwise. The stem, the positional embedding
    /// and the masks (host uploads) stay outside the graph, as its inputs.
    fn encode_padded(&self, x: Tensor, lengths: &[usize]) -> Result<Tensor> {
        let (rows, padded) = (x.dim(0)?, x.dim(1)?);
        let padding = EncoderPadding::new(lengths, padded, x.dtype(), x.device())?;
        let pos = self.positions(padded, x.device())?;
        let graphs = match &self.graphs {
            Ok(graphs) if encoder_graph_shape(rows, padded) == Some((rows, padded)) => graphs,
            _ => return self.layers_forward(x, &pos, Some(&padding)),
        };
        let inputs = [x, pos, padding.key_bias, padding.frames, padding.lengths];
        // The graph keeps a stable copy of each input and an output the size
        // of `x`.
        let tensor_bytes = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        let retained = inputs.iter().map(tensor_bytes).sum::<u64>() + tensor_bytes(&inputs[0]);
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            let [x, pos, key_bias, frames, lengths] = inputs else {
                return Err(Error::InferenceError(format!(
                    "Parakeet encoder graph received {} inputs, expected 5",
                    inputs.len()
                )));
            };
            let padding = EncoderPadding {
                key_bias: key_bias.clone(),
                frames: frames.clone(),
                lengths: lengths.clone(),
            };
            Ok(vec![self.layers_forward(x.clone(), pos, Some(&padding))?])
        };
        graphs
            .run((rows, padded), retained, &inputs, &body)?
            .pop()
            .ok_or_else(|| Error::InferenceError("Parakeet encoder graph returned nothing".into()))
    }

    /// The conformer layers over `x`; `padding` masks a padded batch.
    fn layers_forward(
        &self,
        mut x: Tensor,
        pos: &Tensor,
        padding: Option<&EncoderPadding>,
    ) -> Result<Tensor> {
        for layer in &self.layers {
            x = layer.forward(&x, pos, padding)?;
        }
        Ok(x)
    }
}

/// The padding of a batch of encoder rows zero-padded to one length. A
/// padded frame must reach no real frame: attention adds `key_bias` to its
/// scores, so a padded key's weight is exactly zero, and the convolution
/// module zeroes padded frames before its depthwise convolution, whose
/// kernel would otherwise read them from the last real frames. Padded
/// frames stay finite (every query has a real key), since a non-finite
/// value times a zero weight would still poison a sum.
struct EncoderPadding {
    /// `[B, 1, 1, T]`: 0 on real keys, -1e9 on padding.
    key_bias: Tensor,
    /// `[B, 1, T]`: 1 on real frames, 0 on padding.
    frames: Tensor,
    /// `[B]` U32: each row's real frames, for the fused attention.
    lengths: Tensor,
}

impl EncoderPadding {
    fn new(lengths: &[usize], padded: usize, dtype: DType, device: &Device) -> Result<Self> {
        const MASKED_SCORE: f32 = -1e9;
        if lengths.is_empty() || lengths.iter().any(|len| *len == 0 || *len > padded) {
            return Err(Error::InvalidInput(format!(
                "Parakeet batch rows {lengths:?} do not fit {padded} padded frames"
            )));
        }
        let frames = lengths
            .iter()
            .flat_map(|len| (0..padded).map(move |frame| f32::from(u8::from(frame < *len))))
            .collect::<Vec<_>>();
        let key_bias = frames
            .iter()
            .map(|real| if *real > 0.0 { 0.0 } else { MASKED_SCORE })
            .collect::<Vec<_>>();
        let rows = lengths.len();
        let real = lengths.iter().map(|len| *len as u32).collect::<Vec<_>>();
        Ok(Self {
            key_bias: Tensor::from_vec(key_bias, (rows, 1, 1, padded), device)?.to_dtype(dtype)?,
            frames: Tensor::from_vec(frames, (rows, 1, padded), device)?.to_dtype(dtype)?,
            lengths: Tensor::from_vec(real, rows, device)?,
        })
    }
}

/// The stitched `[1, E, D]` encoder output.
struct ParakeetEncoded {
    encoded: Tensor,
    encoded_len: usize,
    windows: usize,
}

/// Output frames the subsampling stem computes per slice (9.6 s): a slice
/// and its halos read 1,024 feature frames, so its first convolution's
/// output (`[256, 512, 64]` at F32) fills exactly one 32 MiB Metal pool
/// buffer.
const SUBSAMPLING_SLICE_FRAMES: usize = 120;
/// Output frames of input a stem slice reads past each of its edges: the
/// stem's reach is under one output frame, and its zero padding at a slice
/// edge moves inward one position per stride-2 stage.
const SUBSAMPLING_HALO_FRAMES: usize = 4;

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: ProjectionLinear,
    /// The dtype `out` runs in; the convolutions and the output stay F32.
    dtype: DType,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let channels = geometry.subsampling_channels;
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

        let conv0 = mlx::load_conv2d(1, channels, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = channels;
        // Depthwise conv tensors are shaped [out_channels, in_channels/groups, k, k].
        // For groups=channels this second dimension is 1.
        let conv2 = mlx::load_conv2d(1, channels, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(channels, channels, 1, point_cfg, vb.pp("conv.3"))?;
        let conv5 = mlx::load_conv2d(1, channels, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(channels, channels, 1, point_cfg, vb.pp("conv.6"))?;

        let out = projections.linear(mlx::load_linear(
            channels * geometry.subsampled_mel_bins(),
            geometry.dim,
            projections.vb(&vb).pp("out"),
        )?)?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
            dtype: projections.dtype(),
        })
    }

    /// The stem over `features` (`[B, MELS, T]`), `SUBSAMPLING_SLICE_FRAMES`
    /// output frames at a time: its first convolution's output alone is
    /// `[256, T/2, MELS/2]` at F32, 0.34 GB over a full encoder window, and
    /// Metal held its bias add and ReLU beside it. Each output frame `o` reads input frames `8o - 7` to
    /// `8o + 7` through the three stride-2 convolutions, so a slice read with
    /// `SUBSAMPLING_HALO_FRAMES` output frames of input on each side computes
    /// its own frames exactly as the whole sequence does; only the zero
    /// padding at a slice's inner edges differs, and it reaches no kept
    /// frame. Each slice's frames are copied into an exact-size output and
    /// the slice dropped: slices held for one final `Tensor::cat` each
    /// pinned a pooled buffer that best-fit reuse sized up to 64 MiB, growing
    /// the pool by 512 MiB over a window.
    fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let frames = features.dim(2)?;
        let outputs = subsampled_len_3x(frames);
        if outputs <= SUBSAMPLING_SLICE_FRAMES {
            return self.forward_whole(features);
        }
        let mut stitched: Option<Tensor> = None;
        let mut first = 0;
        while first < outputs {
            let last = (first + SUBSAMPLING_SLICE_FRAMES).min(outputs);
            let read_first = first.saturating_sub(SUBSAMPLING_HALO_FRAMES);
            let read_last = (last + SUBSAMPLING_HALO_FRAMES).min(outputs);
            let input_first = read_first * SUBSAMPLING_FACTOR;
            let input_last = (read_last * SUBSAMPLING_FACTOR).min(frames);
            let slice =
                self.forward_whole(&features.narrow(2, input_first, input_last - input_first)?)?;
            let kept = slice
                .narrow(1, first - read_first, last - first)?
                .contiguous()?;
            let output = match stitched.take() {
                Some(output) => output,
                None => exact_zeros(
                    (kept.dim(0)?, outputs, kept.dim(2)?),
                    kept.dtype(),
                    kept.device(),
                )?,
            };
            output.slice_set(&kept, 1, first)?;
            stitched = Some(output);
            first = last;
        }
        stitched.ok_or_else(|| Error::InferenceError("Parakeet stem produced no slices".into()))
    }

    fn forward_whole(&self, features: &Tensor) -> Result<Tensor> {
        // Input features are [B=1, MELS, T]. Convert to [B, 1, T, MELS]
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?;

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = forward_depthwise_conv2d(&self.conv2, &x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = forward_depthwise_conv2d(&self.conv5, &x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        // The layout copy before `out` is also its cast to the projection
        // dtype; the stem output joins the F32 residual stream.
        let (b, c, t, f) = x.dims4()?;
        contiguous_as(&x.transpose(1, 2)?, self.dtype)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?
            .to_dtype(DType::F32)
            .map_err(Error::from)
    }
}

fn forward_depthwise_conv2d(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    let Some(mut x) = metal_kernels::try_depthwise_conv2d(
        x,
        conv.weight(),
        conv.config().padding,
        conv.config().stride,
        conv.config().dilation,
    ) else {
        return conv.forward(x).map_err(Error::from);
    };

    if let Some(bias) = conv.bias() {
        let b = bias.dims1()?;
        x = x.broadcast_add(&bias.reshape((1, b, 1, 1))?)?;
    }

    Ok(x)
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
}

impl ConformerLayer {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let dim = geometry.dim;
        let norm_ff1 = layer_norm(dim, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"), geometry, projections)?;

        let norm_self_att = layer_norm(dim, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"), geometry, projections)?;

        let norm_conv = layer_norm(dim, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"), geometry, projections)?;

        let norm_ff2 = layer_norm(dim, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"), geometry, projections)?;

        let norm_out = layer_norm(dim, 1e-5, vb.pp("norm_out"))?;

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
        })
    }

    /// One layer over `x`; `padding` masks the padded frames of a batch,
    /// and `None` runs an unpadded input.
    fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        padding: Option<&EncoderPadding>,
    ) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn =
            self.self_attn
                .forward(&self.norm_self_att.forward(&residual)?, pos_emb, padding)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(
            &self.norm_conv.forward(&residual)?,
            padding.map(|padding| &padding.frames),
        )?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: ProjectionLinear,
    linear2: ProjectionLinear,
    /// The dtype both linears and the swish between them run in.
    dtype: DType,
}

impl FeedForward {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let vb = projections.vb(&vb);
        let linear1 = projections.linear(mlx::load_linear_no_bias(
            geometry.dim,
            geometry.ff_dim,
            vb.pp("linear1"),
        )?)?;
        let linear2 = projections.linear(mlx::load_linear_no_bias(
            geometry.ff_dim,
            geometry.dim,
            vb.pp("linear2"),
        )?)?;
        Ok(Self {
            linear1,
            linear2,
            dtype: projections.dtype(),
        })
    }

    /// The block over F32 `x`, cast to the block's dtype once on the way in
    /// and back to F32 once on the way out.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(&x.to_dtype(self.dtype)?)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .and_then(|x| x.to_dtype(DType::F32))
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

/// A pointwise convolution of [`ConformerConv`]: dense F32 or weight-only F16
/// over `[B, C, T]`, or the BF16 tier's `[out, in]` linear over the channels
/// of `[B, T, C]`.
enum ConvPointwise {
    Conv(PointwiseConv1d),
    Bf16(Linear),
}

impl ConvPointwise {
    fn as_f16(&self) -> Option<&F16WeightLinear> {
        match self {
            Self::Conv(conv) => conv.as_f16(),
            Self::Bf16(_) => None,
        }
    }

    /// F32 `[B, T, C_in]` to F32 `[B, C_out, T]` (a strided view in the F32
    /// and F16 tiers). The BF16 tier casts in, runs one GEMM, and writes the
    /// channels-first F32 copy in the cast back.
    fn channels_first(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(conv) => conv.forward(&x.transpose(1, 2)?).map_err(Error::from),
            Self::Bf16(linear) => linear
                .forward(&x.to_dtype(DType::BF16)?)?
                .transpose(1, 2)?
                .to_dtype(DType::F32)
                .map_err(Error::from),
        }
    }

    /// F32 `[B, C_in, T]` to F32 `[B, T, C_out]` (a strided view in the F32
    /// and F16 tiers). The BF16 tier writes the channels-last BF16 copy in
    /// its cast in, runs one GEMM and casts back.
    fn channels_last(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(conv) => conv.forward(x)?.transpose(1, 2).map_err(Error::from),
            Self::Bf16(linear) => linear
                .forward(&contiguous_as(&x.transpose(1, 2)?, DType::BF16)?)?
                .to_dtype(DType::F32)
                .map_err(Error::from),
        }
    }
}

struct ConformerConv {
    dim: usize,
    pointwise_conv1: ConvPointwise,
    depthwise_conv: Conv1d,
    folded_depthwise_weight: Tensor,
    folded_depthwise_bias: Tensor,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: ConvPointwise,
}

impl ConformerConv {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let dim = geometry.dim;
        let pointwise_vb = projections.vb(&vb);
        let pointwise_conv1 = projections.pointwise(mlx::load_conv1d_no_bias(
            dim,
            dim * 2,
            1,
            Conv1dConfig::default(),
            pointwise_vb.pp("pointwise_conv1"),
        )?)?;

        let depthwise_conv = mlx::load_conv1d_no_bias(
            dim,
            dim,
            geometry.conv_kernel,
            Conv1dConfig {
                padding: (geometry.conv_kernel - 1) / 2,
                groups: dim,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm = batch_norm(dim, 1e-5, vb.pp("batch_norm"))?;
        let (folded_depthwise_weight, folded_depthwise_bias) =
            fold_depthwise_conv1d_batch_norm(&depthwise_conv, &batch_norm)?;
        // The CUDA depthwise kernel's per-device self-check, once, at load.
        crate::kernels::depthwise_conv::prepare_cuda_kernel(vb.device());

        let pointwise_conv2 = projections.pointwise(mlx::load_conv1d_no_bias(
            dim,
            dim,
            1,
            Conv1dConfig::default(),
            pointwise_vb.pp("pointwise_conv2"),
        )?)?;

        Ok(Self {
            dim,
            pointwise_conv1,
            depthwise_conv,
            folded_depthwise_weight,
            folded_depthwise_bias,
            batch_norm,
            pointwise_conv2,
        })
    }

    /// The convolution module over F32 `[B, T, C]`; `frame_mask`
    /// (`[B, 1, T]`) zeroes a padded batch's padded frames before the
    /// depthwise conv. The GLU, the depthwise conv with its folded BatchNorm
    /// and the swish run in F32 in every tier.
    fn forward(&self, x: &Tensor, frame_mask: Option<&Tensor>) -> Result<Tensor> {
        // [B, T, C] -> [B, 2C, T]
        let mut x = self.pointwise_conv1.channels_first(x)?;
        let x_a = x.i((.., ..self.dim, ..))?;
        let x_b = x.i((.., self.dim.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;
        if let Some(frame_mask) = frame_mask {
            x = x.broadcast_mul(frame_mask)?;
        }

        if let Some(fused) = metal_kernels::try_depthwise_conv1d_bias_swish(
            &x,
            &self.folded_depthwise_weight,
            &self.folded_depthwise_bias,
            self.depthwise_conv.config().padding,
            self.depthwise_conv.config().stride,
            self.depthwise_conv.config().dilation,
        ) {
            x = fused;
        } else {
            let cfg = self.depthwise_conv.config();
            if let Some(depthwise) = metal_kernels::try_depthwise_conv1d(
                &x,
                self.depthwise_conv.weight(),
                cfg.padding,
                cfg.stride,
                cfg.dilation,
            ) {
                x = self.batch_norm.forward_t(&depthwise, false)?;
            } else {
                // Non-Metal: shared shift-and-accumulate op with the folded
                // BatchNorm instead of Candle's per-channel grouped conv.
                x = depthwise_conv1d(
                    &x,
                    &self.folded_depthwise_weight,
                    Some(&self.folded_depthwise_bias),
                    DepthwiseConv1dSpec::symmetric(cfg.padding).with_dilation(cfg.dilation),
                )?;
            }
            x = swish(&x)?;
        }
        // [B, C, T] -> [B, T, C]
        self.pointwise_conv2.channels_last(&x)
    }
}

fn fold_depthwise_conv1d_batch_norm(
    conv: &Conv1d,
    batch_norm: &candle_nn::BatchNorm,
) -> Result<(Tensor, Tensor)> {
    let std = (batch_norm.running_var() + batch_norm.eps())?.sqrt()?;
    let scale = match batch_norm.weight_and_bias() {
        Some((weight, _bias)) => weight.broadcast_div(&std)?,
        None => std.recip()?,
    };
    let folded_bias = match batch_norm.weight_and_bias() {
        Some((_weight, bias)) => {
            bias.broadcast_sub(&scale.broadcast_mul(batch_norm.running_mean())?)?
        }
        None => scale
            .broadcast_mul(batch_norm.running_mean())?
            .affine(-1.0, 0.0)?,
    };

    let channels = scale.dim(0)?;
    let folded_weight = conv
        .weight()
        .broadcast_mul(&scale.reshape((channels, 1, 1))?)?;

    Ok((folded_weight, folded_bias))
}

/// The query, key and value projections of [`RelPosSelfAttention`]: one
/// each (F32 or weight-only F16), or the BF16 tier's single `[3D, D]` GEMM.
enum QkvProjection {
    Split([ProjectionLinear; 3]),
    Fused(Linear),
}

impl QkvProjection {
    /// The projections, when they are separate.
    fn split(&self) -> Option<&[ProjectionLinear; 3]> {
        match self {
            Self::Split(projections) => Some(projections),
            Self::Fused(_) => None,
        }
    }

    /// Q, K and V of F32 `x` (`[B, T, D]`), each a contiguous F32
    /// `[B, H, T, Dh]`. The fused GEMM's head-major F32 copies are its casts
    /// back.
    fn heads(&self, x: &Tensor, heads: usize, head_dim: usize) -> Result<[Tensor; 3]> {
        let (b, t, _) = x.dims3()?;
        let head_major = |x: &Tensor| contiguous_as(&x.transpose(1, 2)?, DType::F32);
        let [q, k, v] = match self {
            Self::Split(projections) => projections.each_ref().map(|projection| {
                head_major(&projection.forward(x)?.reshape((b, t, heads, head_dim))?)
            }),
            Self::Fused(linear) => {
                let qkv = linear
                    .forward(&x.to_dtype(DType::BF16)?)?
                    .reshape((b, t, 3, heads, head_dim))?;
                [0, 1, 2].map(|index| head_major(&qkv.i((.., .., index))?))
            }
        };
        Ok([q?, k?, v?])
    }

    /// Q, K and V of F32 `x` (`[B, T, D]`) as `[B, T, H, Dh]` in the
    /// projection dtype, without head-major copies: the fused GEMM's
    /// columns are strided views of its output.
    fn token_major(&self, x: &Tensor, heads: usize, head_dim: usize) -> Result<[Tensor; 3]> {
        let (b, t, _) = x.dims3()?;
        let [q, k, v] = match self {
            Self::Split(projections) => projections.each_ref().map(|projection| {
                projection
                    .forward(x)?
                    .reshape((b, t, heads, head_dim))
                    .map_err(Error::from)
            }),
            Self::Fused(linear) => {
                let qkv = linear
                    .forward(&x.to_dtype(DType::BF16)?)?
                    .reshape((b, t, 3, heads, head_dim))?;
                [0, 1, 2].map(|index| qkv.i((.., .., index)).map_err(Error::from))
            }
        };
        Ok([q?, k?, v?])
    }
}

/// Whether the conformer attention runs [`relpos_attention::reference`] on
/// the host in place of the CUDA kernel: the in-family tests of the fused
/// path's wiring on a machine without a GPU.
#[cfg(test)]
fn fused_attention_on_the_host() -> bool {
    tests::FUSED_ATTENTION_ON_THE_HOST.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn fused_attention_on_the_host() -> bool {
    false
}

struct RelPosSelfAttention {
    heads: usize,
    head_dim: usize,
    qkv: QkvProjection,
    linear_out: ProjectionLinear,
    linear_pos: ProjectionLinear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    /// The dtype the projections run in. The attention core (`ac + bd`,
    /// `rel_shift`, softmax and the value product) runs in F32.
    dtype: DType,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, projections: Projections) -> Result<Self> {
        let dim = geometry.dim;
        let (heads, head_dim) = (geometry.heads, geometry.head_dim());
        let projection_vb = projections.vb(&vb);
        let read = |name: &str| mlx::load_linear_no_bias(dim, dim, projection_vb.pp(name));
        let (q, k, v) = (read("linear_q")?, read("linear_k")?, read("linear_v")?);
        let qkv = match projections {
            Projections::Bf16(device) => {
                let weight = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?;
                QkvProjection::Fused(bf16_linear(&weight, None, device)?)
            }
            Projections::F32 | Projections::F16(_) => QkvProjection::Split([
                projections.linear(q)?,
                projections.linear(k)?,
                projections.linear(v)?,
            ]),
        };
        let linear_out = projections.linear(read("linear_out")?)?;
        let linear_pos = projections.linear(read("linear_pos")?)?;

        // In Parakeet checkpoints these are direct tensors under self_attn.*.
        let pos_bias_u = vb.get((heads, head_dim), "pos_bias_u")?;
        let pos_bias_v = vb.get((heads, head_dim), "pos_bias_v")?;
        // The kernel's self-check runs at load, before admission prices the
        // encoder from its verdict.
        relpos_attention::prepare_cuda_kernel(vb.device());

        Ok(Self {
            heads,
            head_dim,
            qkv,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            dtype: projections.dtype(),
        })
    }

    /// Attention over F32 `x`; a padded batch's `padding` keeps every query
    /// off its row's padded keys. `pos_emb` is in the projection dtype
    /// ([`ParakeetEncoder::positions`]). On CUDA the fused kernel computes
    /// the core when its self-check passed ([`Self::fused`]).
    fn forward(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        padding: Option<&EncoderPadding>,
    ) -> Result<Tensor> {
        if let Some(out) = self.fused(x, pos_emb, padding)? {
            return Ok(out);
        }
        let key_bias = padding.map(|padding| &padding.key_bias);
        let (b, t, d) = x.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);

        let [q, k, v] = self.qkv.heads(x, heads, head_dim)?; // [B, H, T, Dh]

        let p = self
            .linear_pos
            .forward(&pos_emb.to_dtype(self.dtype)?)?
            .reshape((1, 2 * t - 1, heads, head_dim))?
            .transpose(1, 2)?; // [1, H, 2T-1, Dh]

        let pos_bias_u = self.pos_bias_u.reshape((1, heads, 1, head_dim))?;
        let pos_bias_v = self.pos_bias_v.reshape((1, heads, 1, head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        // Every row of a batch reads the same projection; the matmul takes
        // equal batch dimensions. One copy writes the F32 layout.
        let p_t = p.transpose(2, 3)?;
        let p_t = if b > 1 {
            p_t.broadcast_as((b, heads, head_dim, 2 * t - 1))?
        } else {
            p_t
        };
        let p_t = contiguous_as(&p_t, DType::F32)?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
        let scores = match key_bias {
            Some(key_bias) => scores.broadcast_add(key_bias)?,
            None => scores,
        };
        let scores = if scores.is_contiguous() {
            scores
        } else {
            scores.contiguous()?
        };
        let attn = ops::softmax_last_dim(&scores)?;

        // The token-major copy before `linear_out` is also its cast in.
        let out = attn.matmul(&v)?;
        let out = contiguous_as(&out.transpose(1, 2)?, self.dtype)?.reshape((b, t, d))?;

        self.linear_out
            .forward(&out)
            .and_then(|out| out.to_dtype(DType::F32))
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

impl RelPosSelfAttention {
    /// The attention core as one CUDA launch
    /// ([`relpos_attention::relpos_attention`]): the same F32 core over the
    /// projections' own storage, skipping each row's padded keys, written
    /// token-major in the projection dtype. `None` where the eager core
    /// runs: off CUDA, or with the kernel off for the device.
    fn fused(
        &self,
        x: &Tensor,
        pos_emb: &Tensor,
        padding: Option<&EncoderPadding>,
    ) -> Result<Option<Tensor>> {
        let on_the_host = fused_attention_on_the_host();
        if !on_the_host
            && (!x.device().is_cuda() || relpos_attention::eager_reason(x.device()).is_some())
        {
            return Ok(None);
        }
        let (b, t, d) = x.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);
        let [q, k, v] = self.qkv.token_major(x, heads, head_dim)?;
        let p = self
            .linear_pos
            .forward(&pos_emb.to_dtype(self.dtype)?)?
            .reshape((2 * t - 1, heads, head_dim))?;
        let spec = relpos_attention::RelPosAttention {
            q: &q,
            k: &k,
            v: &v,
            pos: &p,
            position: relpos_attention::RelPosition::TransformerXl {
                bias_u: &self.pos_bias_u,
                bias_v: &self.pos_bias_v,
            },
            band: relpos_attention::KeyBand::Full,
            key_lengths: padding.map_or(relpos_attention::KeyLengths::All, |padding| {
                relpos_attention::KeyLengths::PerRow(&padding.lengths)
            }),
            scale: 1.0 / (head_dim as f64).sqrt(),
            core: DType::F32,
            out_dtype: self.dtype,
        };
        let out = if on_the_host {
            Some(relpos_attention::reference(&spec)?.to_device(x.device())?)
        } else {
            relpos_attention::relpos_attention(&spec)?
        };
        let Some(out) = out else {
            return Ok(None);
        };
        self.linear_out
            .forward(&out.reshape((b, t, d))?)
            .and_then(|out| out.to_dtype(DType::F32))
            .map(Some)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    // x: [B, H, T, 2T-1]
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?; // [B, H, T, 2T]
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

struct Predictor {
    embed: Tensor, // [V+1, H]
    lstm_l0: LstmCell,
    lstm_l1: LstmCell,
    blank_idx: usize,
}

#[derive(Clone)]
pub(crate) struct ParakeetPredictorBatchState {
    h0: Tensor,
    c0: Tensor,
    h1: Tensor,
    c1: Tensor,
}

impl ParakeetPredictorBatchState {
    pub(crate) fn gather_rows(rows: &[&Self]) -> Result<Self> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Parakeet predictor cohort cannot be empty".into(),
            ));
        }
        let gather = |select: fn(&Self) -> &Tensor| {
            let tensors = rows.iter().map(|row| select(row)).collect::<Vec<_>>();
            Tensor::cat(&tensors, 0).map_err(Error::from)
        };
        Ok(Self {
            h0: gather(|state| &state.h0)?,
            c0: gather(|state| &state.c0)?,
            h1: gather(|state| &state.h1)?,
            c1: gather(|state| &state.c1)?,
        })
    }

    pub(crate) fn scatter_rows(&self, rows: &mut [&mut Self]) -> Result<()> {
        let batch = self.h0.dim(0)?;
        if batch == 0
            || rows.len() != batch
            || [self.c0.dim(0)?, self.h1.dim(0)?, self.c1.dim(0)?]
                .into_iter()
                .any(|value| value != batch)
        {
            return Err(Error::InvalidInput(
                "Parakeet predictor cohort scatter has incompatible geometry".into(),
            ));
        }
        let provisional = (0..batch)
            .map(|index| {
                Ok(Self {
                    h0: self.h0.narrow(0, index, 1)?.contiguous()?,
                    c0: self.c0.narrow(0, index, 1)?.contiguous()?,
                    h1: self.h1.narrow(0, index, 1)?.contiguous()?,
                    c1: self.c1.narrow(0, index, 1)?.contiguous()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for (target, value) in rows.iter_mut().zip(provisional) {
            **target = value;
        }
        Ok(())
    }
}

impl Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed = vb.pp("embed").get_unchecked_dtype("weight", DType::F32)?;
        let vocab_plus_blank = embed.dim(0)?;
        let blank_idx = vocab_plus_blank.saturating_sub(1);

        let lstm_l0 = LstmCell::load(vb.pp("dec_rnn.lstm"), 0)?;
        let lstm_l1 = LstmCell::load(vb.pp("dec_rnn.lstm"), 1)?;

        // The device decode's select kernel checks itself at load.
        crate::kernels::transducer_greedy::prepare_cuda_kernel(vb.device());
        Ok(Self {
            embed,
            lstm_l0,
            lstm_l1,
            blank_idx,
        })
    }

    fn initial_state(&self, batch: usize, device: &Device) -> Result<ParakeetPredictorBatchState> {
        let zeros = |dim| Tensor::zeros((batch, dim), DType::F32, device).map_err(Error::from);
        Ok(ParakeetPredictorBatchState {
            h0: zeros(PRED_HIDDEN)?,
            c0: zeros(PRED_HIDDEN)?,
            h1: zeros(PRED_HIDDEN)?,
            c1: zeros(PRED_HIDDEN)?,
        })
    }

    fn step(
        &self,
        label: usize,
        state: &mut ParakeetPredictorBatchState,
        device: &Device,
    ) -> Result<Tensor> {
        self.step_batch(std::slice::from_ref(&label), state, device)
    }

    fn step_batch(
        &self,
        labels: &[usize],
        state: &mut ParakeetPredictorBatchState,
        device: &Device,
    ) -> Result<Tensor> {
        let batch = labels.len();
        if batch == 0
            || [
                state.h0.dim(0),
                state.c0.dim(0),
                state.h1.dim(0),
                state.c1.dim(0),
            ]
            .into_iter()
            .any(|value| value.ok() != Some(batch))
        {
            return Err(Error::InvalidInput(
                "Parakeet predictor labels and recurrent batch state disagree".into(),
            ));
        }
        let rows = labels
            .iter()
            .map(|&label| {
                if label == self.blank_idx {
                    Tensor::zeros((1, PRED_HIDDEN), DType::F32, device).map_err(Error::from)
                } else if label < self.blank_idx {
                    self.embed
                        .i((label, ..))
                        .and_then(|row| row.unsqueeze(0))
                        .map_err(Error::from)
                } else {
                    Err(Error::InvalidInput(format!(
                        "Parakeet predictor label {label} exceeds blank index {}",
                        self.blank_idx
                    )))
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let refs = rows.iter().collect::<Vec<_>>();
        let x = Tensor::cat(&refs, 0)?;

        let (h0, c0) = self.lstm_l0.step(&x, &state.h0, &state.c0)?;
        let (h1, c1) = self.lstm_l1.step(&h0, &state.h1, &state.c1)?;
        // Publish all four recurrent tensors only after both layers succeed.
        state.h0 = h0;
        state.c0 = c0;
        state.h1 = h1;
        state.c1 = c1;

        Ok(state.h1.unsqueeze(1)?) // [B=1, U=1, H]
    }

    fn initialize_physical(
        &self,
        label: usize,
        lease: &mut InvocationTensorLease,
        device: &Device,
    ) -> Result<Tensor> {
        let cursor = lease.arena()?.absolute_cursor();
        if cursor != 0 {
            return Err(Error::InferenceError(
                "Parakeet predictor invocation state was not reset before decode".into(),
            ));
        }
        let mut state = self.initial_state(1, device)?;
        let output = self.step(label, &mut state, device)?;
        commit_parakeet_predictor_state(lease, cursor, &state, &output)?;
        Ok(output)
    }

    fn step_physical(
        &self,
        label: usize,
        lease: &mut InvocationTensorLease,
        device: &Device,
    ) -> Result<Tensor> {
        let snapshot = lease.read_snapshot()?;
        if snapshot.components.len() != 5
            || snapshot
                .components
                .iter()
                .enumerate()
                .any(|(index, value)| value.component != StateComponentId::new((index + 1) as u32))
        {
            return Err(Error::InferenceError(
                "Parakeet physical predictor snapshot has non-canonical components".into(),
            ));
        }
        let mut state = ParakeetPredictorBatchState {
            h0: snapshot.components[0].tensor.clone(),
            c0: snapshot.components[1].tensor.clone(),
            h1: snapshot.components[2].tensor.clone(),
            c1: snapshot.components[3].tensor.clone(),
        };
        let output = self.step(label, &mut state, device)?;
        commit_parakeet_predictor_state(lease, snapshot.absolute_cursor, &state, &output)?;
        Ok(output)
    }
}

fn commit_parakeet_predictor_state(
    lease: &mut InvocationTensorLease,
    expected_cursor: u64,
    state: &ParakeetPredictorBatchState,
    output: &Tensor,
) -> Result<()> {
    let target_cursor = expected_cursor
        .checked_add(1)
        .ok_or_else(|| Error::InferenceError("Parakeet predictor cursor overflow".into()))?;
    let tensors = [
        state.h0.clone(),
        state.c0.clone(),
        state.h1.clone(),
        state.c1.clone(),
        output.clone(),
    ];
    let components = tensors
        .into_iter()
        .enumerate()
        .map(|(index, tensor)| InvocationTensorComponentValue {
            component: StateComponentId::new((index + 1) as u32),
            tensor,
        })
        .collect::<Vec<_>>();
    let declared = components
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let dimensions = if index < 4 {
                vec![
                    ShapeDimensionValue {
                        axis: ShapeAxis::Batch,
                        units: 1,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Hidden,
                        units: PRED_HIDDEN as u64,
                    },
                ]
            } else {
                vec![
                    ShapeDimensionValue {
                        axis: ShapeAxis::Batch,
                        units: 1,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Sequence,
                        units: 1,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Hidden,
                        units: PRED_HIDDEN as u64,
                    },
                ]
            };
            ComponentShapeInstantiation {
                component: value.component,
                dimensions,
            }
        })
        .collect();
    lease.apply_intent(
        &DomainStepIntent {
            domain: physical::PARAKEET_PREDICTOR_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            update: StateUpdateKind::TensorReplace {
                components: declared,
            },
        },
        InvocationTensorUpdateV2::TensorReplace { components },
    )
}

struct LstmCell {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
}

impl LstmCell {
    fn load(vb: VarBuilder, layer: usize) -> Result<Self> {
        let w_ih_name = format!("weight_ih_l{layer}");
        let w_hh_name = format!("weight_hh_l{layer}");
        let b_ih_name = format!("bias_ih_l{layer}");
        let b_hh_name = format!("bias_hh_l{layer}");
        if vb.contains_tensor(&w_ih_name) {
            let w_ih = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_ih_name)?;
            let w_hh = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_hh_name)?;
            let b_ih = vb.get(PRED_HIDDEN * 4, &b_ih_name)?;
            let b_hh = vb.get(PRED_HIDDEN * 4, &b_hh_name)?;
            return Ok(Self {
                w_ih,
                w_hh,
                b_ih,
                b_hh,
            });
        }

        // MLX Parakeet exports LSTM params as dec_rnn.lstm.{layer}.Wx/.Wh/.bias.
        let w_x_name = format!("{layer}.Wx");
        let w_h_name = format!("{layer}.Wh");
        let bias_name = format!("{layer}.bias");
        if !vb.contains_tensor(&w_x_name) || !vb.contains_tensor(&w_h_name) {
            return Err(Error::ModelLoadError(format!(
                "Missing Parakeet LSTM tensors for layer {layer}: expected either weight_ih/weight_hh or {w_x_name}/{w_h_name}"
            )));
        }

        let w_ih = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_x_name)?;
        let w_hh = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_h_name)?;
        let b_ih = if vb.contains_tensor(&bias_name) {
            vb.get(PRED_HIDDEN * 4, &bias_name)?
        } else {
            Tensor::zeros(PRED_HIDDEN * 4, DType::F32, vb.device())?
        };
        let b_hh = Tensor::zeros(PRED_HIDDEN * 4, DType::F32, vb.device())?;

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        })
    }

    fn step(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = x
            .matmul(&self.w_ih.transpose(0, 1)?)?
            .broadcast_add(&self.b_ih.unsqueeze(0)?)?
            .broadcast_add(&h_prev.matmul(&self.w_hh.transpose(0, 1)?)?)?
            .broadcast_add(&self.b_hh.unsqueeze(0)?)?;

        let i = ops::sigmoid(&gates.i((.., 0..PRED_HIDDEN))?)?;
        let f = ops::sigmoid(&gates.i((.., PRED_HIDDEN..(PRED_HIDDEN * 2)))?)?;
        let g = gates
            .i((.., (PRED_HIDDEN * 2)..(PRED_HIDDEN * 3)))?
            .tanh()?;
        let o = ops::sigmoid(&gates.i((.., (PRED_HIDDEN * 3)..))?)?;

        let c = f
            .broadcast_mul(c_prev)?
            .broadcast_add(&i.broadcast_mul(&g)?)?;
        let h = o.broadcast_mul(&c.tanh()?)?;

        Ok((h, c))
    }
}

struct Joint {
    pred: Linear,
    enc: Linear,
    out: Linear,
    num_classes_with_blank: usize,
    num_durations: usize,
}

impl Joint {
    fn load(vb: VarBuilder, enc_hidden: usize, num_classes_with_blank: usize) -> Result<Self> {
        let pred = mlx::load_linear(PRED_HIDDEN, PRED_HIDDEN, vb.pp("pred"))?;
        let enc = mlx::load_linear(enc_hidden, PRED_HIDDEN, vb.pp("enc"))?;

        let out_bias = vb
            .pp("joint_net.2")
            .get_unchecked_dtype("bias", DType::F32)?;
        let out_dim = out_bias.dim(0)?;

        let out = mlx::load_linear(PRED_HIDDEN, out_dim, vb.pp("joint_net.2"))?;

        // TDT adds extra duration logits after token logits.
        let num_durations = out_dim.saturating_sub(num_classes_with_blank);
        if num_durations == 0 {
            return Err(Error::ModelLoadError(format!(
                "Invalid Parakeet joint output size: out_dim={out_dim}, classes_with_blank={num_classes_with_blank}"
            )));
        }

        Ok(Self {
            pred,
            enc,
            out,
            num_classes_with_blank,
            num_durations,
        })
    }

    fn joint_after_projection(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        // f: [B, T, Denc], g: [B, U, Dpred]
        let f = self.enc.forward(f)?;
        let g = self.pred.forward(g)?;

        let inp = f.unsqueeze(2)?.broadcast_add(&g.unsqueeze(1)?)?;
        let inp = inp.relu()?;
        self.out
            .forward(&inp)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn joint_batch_rows(&self, encoded_rows: &Tensor, predictor_rows: &Tensor) -> Result<Tensor> {
        let (batch, _) = encoded_rows.dims2().map_err(|_| {
            Error::InvalidInput("Parakeet batched joint encoder rows must be [B,D]".into())
        })?;
        let (predictor_batch, predictor_steps, _) = predictor_rows.dims3().map_err(|_| {
            Error::InvalidInput("Parakeet batched joint predictor rows must be [B,1,D]".into())
        })?;
        if batch == 0 || predictor_batch != batch || predictor_steps != 1 {
            return Err(Error::InvalidInput(
                "Parakeet batched joint rows have incompatible batch geometry".into(),
            ));
        }
        self.joint_after_projection(&encoded_rows.unsqueeze(1)?, predictor_rows)?
            .squeeze(2)?
            .squeeze(1)
            .map_err(Error::from)
    }
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

/// The TDT label (argmax over `logits[..token_end]`) and duration index
/// (argmax over the rest) of one joint output, read back together: one
/// device-to-host sync per joint step instead of two.
fn tdt_label_and_duration(logits: &Tensor, token_end: usize) -> Result<(usize, usize)> {
    let width = logits.dims1()?;
    if token_end == 0 || width <= token_end {
        return Err(Error::InferenceError(format!(
            "Parakeet TDT joint output of width {width} has no token or duration logits after {token_end}"
        )));
    }
    let selected =
        tdt_batch_labels_and_durations(&logits.unsqueeze(0)?, token_end, width - token_end)?;
    Ok(selected[0])
}

/// Per-row TDT label and duration indices of a `(rows, width)` joint output
/// with one readback for all rows.
fn tdt_batch_labels_and_durations(
    logits: &Tensor,
    token_end: usize,
    num_durations: usize,
) -> Result<Vec<(usize, usize)>> {
    let (rows, width) = logits.dims2()?;
    if token_end == 0 || num_durations == 0 || width < token_end + num_durations {
        return Err(Error::InferenceError(format!(
            "Parakeet TDT joint output of width {width} cannot hold {token_end} token and {num_durations} duration logits"
        )));
    }
    if logits.device().is_cuda() {
        // Ties to the lowest id and NaN never picked, as the device decode's
        // select step and the CPU's argmax pick (Candle's CUDA argmax breaks
        // ties by its strided reduction).
        use crate::kernels::greedy::greedy_packed;
        let labels = greedy_packed(&logits.narrow(1, 0, token_end)?.contiguous()?)?;
        let durations = greedy_packed(&logits.narrow(1, token_end, num_durations)?.contiguous()?)?;
        crate::models::shared::telemetry::record_host_read(DType::U32, 4 * rows);
        let picks = Tensor::cat(&[labels, durations], 1)?.to_vec2::<u32>()?;
        return Ok(picks
            .iter()
            .map(|pick| (pick[0] as usize, pick[2] as usize))
            .collect());
    }
    let labels = logits.narrow(1, 0, token_end)?.argmax(D::Minus1)?;
    let durations = logits
        .narrow(1, token_end, num_durations)?
        .argmax(D::Minus1)?;
    let selected = Tensor::stack(&[labels, durations], 0)?.to_dtype(DType::U32)?;
    crate::models::shared::telemetry::record_dtype_cast();
    crate::models::shared::telemetry::record_host_read(DType::U32, 2 * rows);
    let selected = selected.to_vec2::<u32>()?;
    (0..rows)
        .map(|row| Ok((selected[0][row] as usize, selected[1][row] as usize)))
        .collect()
}

fn resampled_len(len: usize, src_rate: u32, dst_rate: u32) -> usize {
    if src_rate == dst_rate || len < 2 {
        return len;
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    ((len as f64) * ratio).round().max(1.0) as usize
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = resampled_len(audio.len(), src_rate, dst_rate);

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let sample = audio[left] * (1.0 - frac) + audio[right] * frac;
        out.push(sample);
    }
    out
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::{
        build_rel_positional_embedding, encoder_window_row_bytes, parakeet_encode_row_bytes,
        parakeet_encoded_len_for_samples, parakeet_max_emittable_tokens,
        parakeet_preparation_bytes, parakeet_tensor_is_f16_projection, plan_encoder_windows,
        subsampled_len_3x, tdt_word_timestamps, window_batch_rows, EncoderGeometry, EncoderWindow,
        EncoderWindowing, Joint, LstmCell, ParakeetAsrModel, ParakeetDecoder, ParakeetEncoded,
        ParakeetEncoder, ParakeetEncoderBatchRow, ParakeetHostFeatures, ParakeetLogMel,
        ParakeetNetwork, ParakeetPreparedEncoderArtifact, ParakeetPreprocessor,
        ParakeetRetainedDecodeBatchRow, ParakeetRetainedDecodeStep, ParakeetTimings, Predictor,
        TdtTokenFrame, DEFAULT_MAX_SYMBOLS, PRED_HIDDEN,
    };
    use super::{
        encoder_batch_shape, encoder_graph_shape, encoder_graphs_requested,
        encoder_window_batch_bytes, parakeet_encode_row_elements,
        parakeet_encoder_graph_device_bytes, ENCODER_GRAPH_LIMITS, ENCODER_GRAPH_ROWS,
        PARAKEET_ENCODER_GRAPHS_ENV,
    };
    use super::{encoder_window_batch_bytes_with, EncoderPadding};
    use super::{
        parakeet_cuda_projection_precision, parakeet_tensor_is_bf16_projection,
        resolve_parakeet_projection_precision, ConvPointwise, Projections, QkvProjection,
        PARAKEET_ASR_DTYPE_ENV,
    };
    use super::{tdt_batch_labels_and_durations, tdt_label_and_duration};

    thread_local! {
        /// Runs the conformer attention through the fused kernel's host
        /// reference (see `fused_attention_on_the_host`).
        pub(super) static FUSED_ATTENTION_ON_THE_HOST: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
        /// Runs the retained decode's dispatches with the select step on
        /// the host (see `device_decode::host_select`).
        pub(super) static DEVICE_DECODE_ON_THE_HOST: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }
    use crate::backends::{BackendKind, DTypeSelection};
    use crate::error::{Error, Result};
    use crate::model::ModelVariant;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, VarBuilder};

    use crate::kernels::f16_weight_linear::{ProjectionLinear, WeightOnlyF16Load};
    use crate::kernels::numeric_check::{host_f32, relative_l2};

    /// Small enough to run on CPU in milliseconds; two layers keep conv edge
    /// effects within 4 × 2 + 1 frames of a window boundary.
    const TINY: EncoderGeometry = EncoderGeometry {
        mel_bins: 128,
        subsampling_channels: 4,
        layers: 2,
        dim: 32,
        heads: 4,
        ff_dim: 64,
        conv_kernel: 9,
    };

    /// xorshift64: unstructured values, so a wrong index cannot cancel out.
    struct XorShift(u64);

    impl XorShift {
        fn next_unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }

        fn values(&mut self, len: usize, scale: f32, offset: f32) -> Vec<f32> {
            (0..len)
                .map(|_| offset + scale * self.next_unit())
                .collect()
        }
    }

    struct RandomWeights<'a> {
        rng: XorShift,
        device: &'a Device,
        tensors: HashMap<String, Tensor>,
    }

    impl RandomWeights<'_> {
        fn put(&mut self, name: String, shape: &[usize], scale: f32, offset: f32) {
            let values = self.rng.values(shape.iter().product(), scale, offset);
            let tensor = Tensor::from_vec(values, shape, self.device).unwrap();
            self.tensors.insert(name, tensor);
        }

        fn put_weight(&mut self, name: String, shape: &[usize], fan_in: usize) {
            self.put(name, shape, 1.0 / (fan_in as f32).sqrt(), 0.0);
        }
    }

    /// The subsampling stem run by slices returns the frames of the
    /// whole-sequence stem, for lengths that leave a short last slice. A
    /// slice changes only how many rows each matrix product runs over, so
    /// the frames agree within F32 rounding: the CPU product blocks its sums
    /// by problem size, instruction set and thread count (3e-8 apart on CI,
    /// bit for bit on the dev Mac).
    #[test]
    fn sliced_subsampling_matches_the_whole_sequence() {
        let device = Device::Cpu;
        let stem = super::ConvSubsamplingDw::load(
            random_encoder_vb(TINY, 0x5EED, &device).pp("pre_encode"),
            TINY,
            Projections::F32,
        )
        .unwrap();
        for frames in [8 * super::SUBSAMPLING_SLICE_FRAMES * 2 + 3, 5_001] {
            let features = Tensor::from_vec(
                XorShift(frames as u64).values(TINY.mel_bins * frames, 1.0, 0.0),
                (1, TINY.mel_bins, frames),
                &device,
            )
            .unwrap();
            let whole = stem.forward_whole(&features).unwrap();
            let sliced = stem.forward(&features).unwrap();
            assert_eq!(sliced.dims(), whole.dims(), "{frames}");
            let scale = whole
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let difference = (sliced - whole)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                difference <= scale * 1e-6,
                "{frames}: {difference} of {scale}"
            );
        }
    }

    /// A NeMo-layout encoder checkpoint with random weights, loaded through
    /// the production loaders.
    fn random_encoder(geometry: EncoderGeometry, seed: u64, device: &Device) -> ParakeetEncoder {
        ParakeetEncoder::load(
            random_encoder_vb(geometry, seed, device),
            geometry,
            Projections::F32,
        )
        .unwrap()
    }

    /// The random encoder checkpoint of [`random_encoder`] as a VarBuilder.
    fn random_encoder_vb(
        geometry: EncoderGeometry,
        seed: u64,
        device: &Device,
    ) -> VarBuilder<'static> {
        VarBuilder::from_tensors(
            random_encoder_tensors(geometry, seed, device),
            DType::F32,
            device,
        )
    }

    fn random_encoder_tensors(
        geometry: EncoderGeometry,
        seed: u64,
        device: &Device,
    ) -> HashMap<String, Tensor> {
        let (c, d, ff, k) = (
            geometry.subsampling_channels,
            geometry.dim,
            geometry.ff_dim,
            geometry.conv_kernel,
        );
        let mut weights = RandomWeights {
            rng: XorShift(seed),
            device,
            tensors: HashMap::new(),
        };
        for (conv, shape) in [
            ("conv.0", [c, 1, 3, 3]),
            ("conv.2", [c, 1, 3, 3]),
            ("conv.3", [c, c, 1, 1]),
            ("conv.5", [c, 1, 3, 3]),
            ("conv.6", [c, c, 1, 1]),
        ] {
            let fan_in = shape[1] * shape[2] * shape[3];
            weights.put_weight(format!("pre_encode.{conv}.weight"), &shape, fan_in);
            weights.put(format!("pre_encode.{conv}.bias"), &[c], 0.1, 0.0);
        }
        let subsampled = c * geometry.subsampled_mel_bins();
        weights.put_weight("pre_encode.out.weight".into(), &[d, subsampled], subsampled);
        weights.put("pre_encode.out.bias".into(), &[d], 0.1, 0.0);
        for layer in 0..geometry.layers {
            let prefix = format!("layers.{layer}");
            for norm in [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "norm_feed_forward2",
                "norm_out",
            ] {
                weights.put(format!("{prefix}.{norm}.weight"), &[d], 0.1, 1.0);
                weights.put(format!("{prefix}.{norm}.bias"), &[d], 0.1, 0.0);
            }
            for ffn in ["feed_forward1", "feed_forward2"] {
                weights.put_weight(format!("{prefix}.{ffn}.linear1.weight"), &[ff, d], d);
                weights.put_weight(format!("{prefix}.{ffn}.linear2.weight"), &[d, ff], ff);
            }
            let attention = format!("{prefix}.self_attn");
            for projection in [
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_out",
                "linear_pos",
            ] {
                weights.put_weight(format!("{attention}.{projection}.weight"), &[d, d], d);
            }
            for bias in ["pos_bias_u", "pos_bias_v"] {
                let shape = [geometry.heads, geometry.head_dim()];
                weights.put(format!("{attention}.{bias}"), &shape, 0.1, 0.0);
            }
            let conv = format!("{prefix}.conv");
            weights.put_weight(format!("{conv}.pointwise_conv1.weight"), &[2 * d, d, 1], d);
            weights.put_weight(format!("{conv}.depthwise_conv.weight"), &[d, 1, k], k);
            weights.put(format!("{conv}.batch_norm.weight"), &[d], 0.1, 1.0);
            weights.put(format!("{conv}.batch_norm.bias"), &[d], 0.1, 0.0);
            weights.put(format!("{conv}.batch_norm.running_mean"), &[d], 0.1, 0.0);
            weights.put(format!("{conv}.batch_norm.running_var"), &[d], 0.25, 1.0);
            weights.put_weight(format!("{conv}.pointwise_conv2.weight"), &[d, d, 1], d);
        }
        weights.tensors
    }

    fn random_log_mel(frames: usize, valid_frames: usize, seed: u64) -> ParakeetLogMel {
        let values = XorShift(seed).values(TINY.mel_bins * frames, 4.0, -2.0);
        ParakeetLogMel::from_parts(values, TINY.mel_bins, frames, valid_frames)
    }

    /// Today's whole-utterance encode: every log-mel column through one
    /// pre-encode and one attention pass. `truncate_first` drops the STFT
    /// padding frame before attention (the windowed encoder's behaviour);
    /// otherwise it is dropped afterwards, as before windowing.
    fn whole_utterance_encode(
        encoder: &ParakeetEncoder,
        features: &Tensor,
        encoded_len: usize,
        truncate_first: bool,
    ) -> Tensor {
        let mut x = encoder.pre_encode.forward(features).unwrap();
        if truncate_first {
            x = x.narrow(1, 0, encoded_len).unwrap();
        }
        let pos =
            build_rel_positional_embedding(x.dim(1).unwrap(), encoder.geometry.dim, x.device())
                .unwrap();
        for layer in &encoder.layers {
            x = layer.forward(&x, &pos, None).unwrap();
        }
        x.narrow(1, 0, encoded_len).unwrap()
    }

    fn encode_windows(
        encoder: &ParakeetEncoder,
        feature_frames: usize,
        encoded_len: usize,
        windowing: EncoderWindowing,
        window_features: &mut dyn FnMut(std::ops::Range<usize>) -> Result<Tensor>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetEncoded> {
        encode_windows_stacked(
            encoder,
            feature_frames,
            encoded_len,
            windowing,
            1,
            window_features,
            checkpoint,
        )
    }

    /// [`encode_windows`] with up to `window_rows` windows per batch.
    fn encode_windows_stacked(
        encoder: &ParakeetEncoder,
        feature_frames: usize,
        encoded_len: usize,
        windowing: EncoderWindowing,
        window_rows: usize,
        window_features: &mut dyn FnMut(std::ops::Range<usize>) -> Result<Tensor>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetEncoded> {
        encoder.encode_windows(
            feature_frames,
            encoded_len,
            windowing,
            window_rows,
            window_features,
            &mut ParakeetTimings::default(),
            checkpoint,
        )
    }

    /// Largest absolute difference; NaN or infinity counts as infinitely far.
    /// Batched vs solo encodes on CPU F32, as a fraction of the row's scale.
    const BATCHED_ENCODE_TOLERANCE: f32 = 4e-6;
    /// BF16-tier encodes on CPU, as relative L2 against the reference.
    const BF16_ENCODE_RELATIVE_L2: f64 = 2e-2;

    fn max_abs_diff(left: &Tensor, right: &Tensor) -> f32 {
        assert_eq!(left.dims(), right.dims());
        let values = |tensor: &Tensor| tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        values(left)
            .into_iter()
            .zip(values(right))
            .map(|(left, right)| {
                let difference = (left - right).abs();
                if difference.is_finite() {
                    difference
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    /// The non-Metal conformer conv (shared shift-and-accumulate depthwise op
    /// with the folded BatchNorm) matches Candle's per-channel grouped conv
    /// followed by the unfolded BatchNorm.
    #[test]
    fn conformer_conv_shared_depthwise_matches_candle_grouped_conv() {
        use candle_nn::{Module, ModuleT};

        let device = Device::Cpu;
        let encoder = random_encoder(TINY, 0x51ED_2701, &device);
        let values = XorShift(0xC0FF_EE11).values(2 * 37 * TINY.dim, 1.0, 0.0);
        let x = Tensor::from_vec(values, (2, 37, TINY.dim), &device).unwrap();
        for layer in &encoder.layers {
            let conv = &layer.conv;
            let calls = crate::kernels::depthwise_conv::depthwise_conv1d_calls();
            let actual = conv.forward(&x, None).unwrap();
            assert!(crate::kernels::depthwise_conv::depthwise_conv1d_calls() > calls);

            let h = conv.pointwise_conv1.channels_first(&x).unwrap();
            let gate = h.narrow(1, 0, TINY.dim).unwrap();
            let value = h.narrow(1, TINY.dim, TINY.dim).unwrap();
            let h = gate
                .broadcast_mul(&candle_nn::ops::sigmoid(&value).unwrap())
                .unwrap();
            let h = conv.depthwise_conv.forward(&h).unwrap();
            let h = conv.batch_norm.forward_t(&h, false).unwrap();
            let h = super::swish(&h).unwrap();
            let expected = conv.pointwise_conv2.channels_last(&h).unwrap();
            let diff = max_abs_diff(&expected, &actual);
            assert!(diff <= 1e-5, "max |diff| {diff}");
        }
    }

    fn zero_lstm(device: &Device) -> LstmCell {
        LstmCell {
            w_ih: Tensor::zeros((PRED_HIDDEN * 4, PRED_HIDDEN), DType::F32, device).unwrap(),
            w_hh: Tensor::zeros((PRED_HIDDEN * 4, PRED_HIDDEN), DType::F32, device).unwrap(),
            b_ih: Tensor::zeros(PRED_HIDDEN * 4, DType::F32, device).unwrap(),
            b_hh: Tensor::zeros(PRED_HIDDEN * 4, DType::F32, device).unwrap(),
        }
    }

    fn zero_predictor(device: &Device) -> Predictor {
        Predictor {
            embed: Tensor::zeros((3, PRED_HIDDEN), DType::F32, device).unwrap(),
            lstm_l0: zero_lstm(device),
            lstm_l1: zero_lstm(device),
            blank_idx: 2,
        }
    }

    /// A model whose joint always emits label 0 with duration 0, so every
    /// encoder frame emits exactly `max_symbols` labels. The retained decode
    /// only reads the prepared artifact, never the encoder.
    fn forced_label_model(device: &Device) -> ParakeetAsrModel {
        let predictor = zero_predictor(device);
        let joint = Joint {
            pred: Linear::new(
                Tensor::zeros((PRED_HIDDEN, PRED_HIDDEN), DType::F32, device).unwrap(),
                None,
            ),
            enc: Linear::new(
                Tensor::zeros((PRED_HIDDEN, TINY.dim), DType::F32, device).unwrap(),
                None,
            ),
            // Labels [0, 1, blank] then durations [0, 1].
            out: Linear::new(
                Tensor::zeros((5, PRED_HIDDEN), DType::F32, device).unwrap(),
                Some(Tensor::new(&[4f32, 1.0, 0.0, 3.0, 0.0], device).unwrap()),
            ),
            num_classes_with_blank: 3,
            num_durations: 2,
        };
        let blank_idx = predictor.blank_idx;
        let num_durations = joint.num_durations;
        ParakeetAsrModel {
            variant: ModelVariant::ParakeetTdt06BV3,
            _artifacts: None,
            decoder: ParakeetDecoder::Vocab(vec!["\u{2581}a".into(), "\u{2581}b".into()]),
            preprocessor: ParakeetPreprocessor::for_tests(device.clone()),
            network: ParakeetNetwork {
                encoder: random_encoder(TINY, 1, device),
                predictor,
                joint,
                blank_idx,
                num_durations,
            },
            blank_idx,
            num_durations,
            max_symbols: DEFAULT_MAX_SYMBOLS,
            encoder_windowing: EncoderWindowing::V3,
            weight_precision: WeightOnlyF16Load::dense("test"),
        }
    }

    fn zero_artifact(device: &Device, encoded_len: usize) -> ParakeetPreparedEncoderArtifact {
        ParakeetPreparedEncoderArtifact {
            encoded: Tensor::zeros((1, encoded_len, TINY.dim), DType::F32, device).unwrap(),
            encoded_len,
            input_samples: encoded_len * 1280,
            input_sample_rate: 16_000,
            resampled_samples: encoded_len * 1280,
            feature_frames: encoded_len * 8,
            language: None,
            windowing: EncoderWindowing::V3,
            encoder_windows: 1,
            log_mel_ms: 0.0,
            encode_ms: 0.0,
        }
    }

    fn decode_until_finished(
        model: &ParakeetAsrModel,
        artifact: &ParakeetPreparedEncoderArtifact,
        max_tokens: usize,
    ) -> (Vec<ParakeetRetainedDecodeStep>, serde_json::Value) {
        let mut state = model.start_retained_decode().unwrap();
        let mut steps = Vec::new();
        while steps
            .last()
            .is_none_or(|step: &ParakeetRetainedDecodeStep| !step.finished)
        {
            assert!(steps.len() <= 1_000, "retained decode never finished");
            let mut rows = [ParakeetRetainedDecodeBatchRow {
                artifact,
                state: &mut state,
                max_tokens,
                quota: 1,
            }];
            steps.extend(model.retained_decode_step_batch(&mut rows).unwrap());
        }
        let diagnostics = model
            .retained_decode_output(artifact, &state)
            .diagnostics
            .unwrap();
        (steps, diagnostics)
    }

    /// A TDT model whose picks depend on the encoder frames and the
    /// predictor state: random predictor, joint and embeddings, labels
    /// `[0, 4)`, blank 4 and durations `0..5`.
    fn random_decoder_model(device: &Device, seed: u64) -> ParakeetAsrModel {
        let mut rng = XorShift(seed);
        let mut tensor = |shape: &[usize], scale: f32| {
            let len = shape.iter().product();
            Tensor::from_vec(rng.values(len, scale, 0.0), shape, device).unwrap()
        };
        let mut lstm = || LstmCell {
            w_ih: tensor(&[PRED_HIDDEN * 4, PRED_HIDDEN], 0.05),
            w_hh: tensor(&[PRED_HIDDEN * 4, PRED_HIDDEN], 0.05),
            b_ih: tensor(&[PRED_HIDDEN * 4], 0.1),
            b_hh: tensor(&[PRED_HIDDEN * 4], 0.1),
        };
        let (lstm_l0, lstm_l1) = (lstm(), lstm());
        let predictor = Predictor {
            embed: tensor(&[5, PRED_HIDDEN], 1.0),
            lstm_l0,
            lstm_l1,
            blank_idx: 4,
        };
        let joint = Joint {
            pred: Linear::new(tensor(&[PRED_HIDDEN, PRED_HIDDEN], 0.05), None),
            enc: Linear::new(tensor(&[PRED_HIDDEN, TINY.dim], 0.3), None),
            out: Linear::new(tensor(&[10, PRED_HIDDEN], 0.1), Some(tensor(&[10], 0.5))),
            num_classes_with_blank: 5,
            num_durations: 5,
        };
        let mut model = forced_label_model(device);
        model.network.predictor = predictor;
        model.network.joint = joint;
        model.network.blank_idx = 4;
        model.network.num_durations = 5;
        model.blank_idx = 4;
        model.num_durations = 5;
        model.decoder = ParakeetDecoder::Vocab(
            ["\u{2581}a", "\u{2581}b", "c", "\u{2581}d"]
                .map(String::from)
                .to_vec(),
        );
        model
    }

    /// Decodes `artifacts` as one retained cohort (finished rows leave it)
    /// until every row finishes: each row's tokens, frames, text and joint
    /// and blank steps.
    fn decode_cohort(
        model: &ParakeetAsrModel,
        artifacts: &[ParakeetPreparedEncoderArtifact],
        max_tokens: &[usize],
        quota: usize,
    ) -> Vec<(Vec<usize>, Vec<TdtTokenFrame>, String, usize, usize)> {
        let mut states = artifacts
            .iter()
            .map(|_| model.start_retained_decode().unwrap())
            .collect::<Vec<_>>();
        for _ in 0..10_000 {
            let mut rows = states
                .iter_mut()
                .zip(artifacts)
                .zip(max_tokens)
                .filter(|((state, _), _)| !state.finished)
                .map(
                    |((state, artifact), max_tokens)| ParakeetRetainedDecodeBatchRow {
                        artifact,
                        state,
                        max_tokens: *max_tokens,
                        quota,
                    },
                )
                .collect::<Vec<_>>();
            if rows.is_empty() {
                break;
            }
            let steps = model.retained_decode_step_batch(&mut rows).unwrap();
            for step in &steps {
                assert!(step.tokens_generated <= 10_000);
            }
        }
        states
            .into_iter()
            .map(|state| {
                assert!(state.finished, "a row never finished");
                (
                    state.token_ids,
                    state.token_frames,
                    state.assembled,
                    state.counters.tdt_joint_steps,
                    state.counters.tdt_blank_steps,
                )
            })
            .collect()
    }

    /// The dispatches of many joint steps decode every row of a cohort as
    /// the eager step does, with the select step on the host: the same
    /// tokens, frames, text and joint and blank steps, across window
    /// boundaries (a 70-frame row; windows hold 64), a token budget, rows finishing at
    /// different steps, and a quota of several tokens per engine step.
    #[test]
    fn device_decode_dispatches_match_the_eager_retained_decode() {
        let device = Device::Cpu;
        let model = random_decoder_model(&device, 0xDEC0_DE01);
        let mut rng = XorShift(0xA77_1FAC7);
        let artifacts = [12usize, 70, 5, 20]
            .map(|len| {
                let mut artifact = zero_artifact(&device, len);
                artifact.encoded = Tensor::from_vec(
                    rng.values(len * TINY.dim, 2.0, 0.0),
                    (1, len, TINY.dim),
                    &device,
                )
                .unwrap();
                artifact
            })
            .to_vec();
        let max_tokens = [1_000, 1_000, 1_000, 3];
        let eager = decode_cohort(&model, &artifacts, &max_tokens, 1);
        assert!(eager.iter().map(|row| row.0.len()).sum::<usize>() > 10);
        assert!(eager.iter().any(|row| row.4 > 0), "no blanks");
        assert_eq!(eager[3].0.len(), 3, "the budget stops the last row");
        for quota in [1, 3] {
            DEVICE_DECODE_ON_THE_HOST.with(|flag| flag.set(true));
            let dispatched = decode_cohort(&model, &artifacts, &max_tokens, quota);
            DEVICE_DECODE_ON_THE_HOST.with(|flag| flag.set(false));
            for (row, (eager, dispatched)) in eager.iter().zip(&dispatched).enumerate() {
                assert_eq!(eager, dispatched, "row {row}, quota {quota}");
            }
        }
    }

    /// Drives a retained cohort as the engine's handler does, at `quota`
    /// tokens per row and step: each step's new tokens (at most the quota),
    /// its delta appended to the row's stream (which must equal the step's
    /// text), and, at `cancel = (row, step)`, that row restored from its
    /// checkpoint as the handler's rollback restores a cancelled row (it must
    /// equal the checkpoint; the row then leaves the cohort). Each row's
    /// tokens, streamed text, final text and word timestamps.
    fn run_like_the_handler(
        model: &ParakeetAsrModel,
        artifacts: &[ParakeetPreparedEncoderArtifact],
        max_tokens: &[usize],
        quota: usize,
        cancel: Option<(usize, usize)>,
    ) -> Vec<(Vec<usize>, String, String, serde_json::Value)> {
        let mut states = artifacts
            .iter()
            .map(|_| model.start_retained_decode().unwrap())
            .collect::<Vec<_>>();
        let mut streamed = vec![String::new(); artifacts.len()];
        let mut gone = vec![false; artifacts.len()];
        for step in 0..10_000 {
            let live = (0..states.len())
                .filter(|row| !states[*row].finished && !gone[*row])
                .collect::<Vec<_>>();
            if live.is_empty() {
                break;
            }
            let checkpoints = live
                .iter()
                .map(|row| states[*row].clone())
                .collect::<Vec<_>>();
            let before = live
                .iter()
                .map(|row| states[*row].token_ids.len())
                .collect::<Vec<_>>();
            let mut rows = states
                .iter_mut()
                .enumerate()
                .filter(|(row, _)| live.contains(row))
                .map(|(row, state)| ParakeetRetainedDecodeBatchRow {
                    artifact: &artifacts[row],
                    state,
                    max_tokens: max_tokens[row],
                    quota,
                })
                .collect::<Vec<_>>();
            let steps = model.retained_decode_step_batch(&mut rows).unwrap();
            drop(rows);
            for (((row, step_out), before), checkpoint) in
                live.iter().zip(steps).zip(before).zip(checkpoints)
            {
                if cancel == Some((*row, step)) {
                    states[*row] = checkpoint.clone();
                    assert_eq!(states[*row].token_ids, checkpoint.token_ids);
                    assert_eq!(states[*row].cursor, checkpoint.cursor);
                    gone[*row] = true;
                    continue;
                }
                assert!(step_out.tokens_generated <= before + quota);
                assert_eq!(step_out.tokens_generated, states[*row].token_ids.len());
                streamed[*row].push_str(&step_out.delta);
                assert_eq!(step_out.text, streamed[*row]);
            }
        }
        states
            .iter()
            .zip(artifacts)
            .zip(streamed)
            .map(|((state, artifact), streamed)| {
                let output = model.retained_decode_output(artifact, state);
                let words = output.diagnostics.unwrap()["timestamp_words"].clone();
                (state.token_ids.clone(), streamed, output.text, words)
            })
            .collect()
    }

    /// The engine contract of several tokens per row and step: a cohort at a
    /// quota of four (the device dispatches, select step on the host)
    /// decodes the same tokens, streamed text, final text and word
    /// timestamps as at one token per step (the eager step); a row whose
    /// budget ends mid-quota finishes there; and a row cancelled mid-quota
    /// is restored to its checkpoint without changing the other rows.
    #[test]
    fn quanta_keep_tokens_text_and_timestamps() {
        let device = Device::Cpu;
        let model = random_decoder_model(&device, 0xDEC0_DE01);
        let mut rng = XorShift(0xA77_1FAC7);
        let artifacts = [12usize, 40, 6]
            .map(|len| {
                let mut artifact = zero_artifact(&device, len);
                let values = rng.values(len * TINY.dim, 2.0, 0.0);
                artifact.encoded = Tensor::from_vec(values, (1, len, TINY.dim), &device).unwrap();
                artifact
            })
            .to_vec();
        let max_tokens = [1_000, 5, 1_000];
        let one = run_like_the_handler(&model, &artifacts, &max_tokens, 1, None);
        assert_eq!(one[1].0.len(), 5, "the budget ends the second row");
        let lengths = one.iter().map(|row| row.0.len()).collect::<Vec<_>>();
        assert!(lengths.iter().sum::<usize>() > 8, "{lengths:?}");
        // Cancel the first row with more than one quantum of tokens.
        let row = (0..one.len()).find(|row| one[*row].0.len() > 4).unwrap();
        DEVICE_DECODE_ON_THE_HOST.with(|flag| flag.set(true));
        let four = run_like_the_handler(&model, &artifacts, &max_tokens, 4, None);
        let cancelled = run_like_the_handler(&model, &artifacts, &max_tokens, 4, Some((row, 1)));
        DEVICE_DECODE_ON_THE_HOST.with(|flag| flag.set(false));
        assert_eq!(one, four);
        // The cancelled row keeps its first step's tokens; the others are
        // as before.
        for other in (0..one.len()).filter(|other| *other != row) {
            assert_eq!(cancelled[other], four[other], "row {other}");
        }
        assert!(cancelled[row].0.len() <= 4 && four[row].0.starts_with(&cancelled[row].0));
    }

    #[test]
    fn parakeet_encoded_len_follows_front_end_geometry() {
        // test.wav: 3.6 s -> 360 valid feature frames -> 45 encoder frames.
        assert_eq!(parakeet_encoded_len_for_samples(57_600, 16_000), 45);
        assert_eq!(parakeet_encoded_len_for_samples(172_800, 48_000), 45);
        // 29.7 s -> 2970 feature frames -> ceil(2970 / 8) frames.
        assert_eq!(parakeet_encoded_len_for_samples(475_200, 16_000), 372);
        assert_eq!(parakeet_encoded_len_for_samples(159, 16_000), 0);
        assert_eq!(parakeet_encoded_len_for_samples(160, 16_000), 1);

        assert_eq!(
            parakeet_max_emittable_tokens(475_200, 16_000, DEFAULT_MAX_SYMBOLS).unwrap(),
            3_720
        );
        assert!(matches!(
            parakeet_max_emittable_tokens(159, 16_000, DEFAULT_MAX_SYMBOLS),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            parakeet_max_emittable_tokens(16_000, 0, DEFAULT_MAX_SYMBOLS),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn retained_decode_finishes_at_max_tokens_with_stop_reason() {
        let device = Device::Cpu;
        let model = forced_label_model(&device);
        let artifact = zero_artifact(&device, 4);

        let (steps, diagnostics) = decode_until_finished(&model, &artifact, 5);

        assert_eq!(steps.len(), 5);
        assert!(steps[..4].iter().all(|step| !step.finished));
        assert_eq!(steps[4].tokens_generated, 5);
        assert_eq!(diagnostics["stop_reason"], "max_tokens");
        assert_eq!(diagnostics["tdt_emitted_tokens"], 5);
    }

    #[test]
    fn retained_decode_at_emittable_bound_ends_without_cap() {
        let device = Device::Cpu;
        let model = forced_label_model(&device);
        let artifact = zero_artifact(&device, 4);
        let bound = artifact.encoded_len * DEFAULT_MAX_SYMBOLS;

        let (steps, diagnostics) = decode_until_finished(&model, &artifact, bound);

        // Every frame emits max_symbols labels, so the bound is exactly the
        // transcript: the decode ends there without reporting a cap.
        assert_eq!(steps.last().unwrap().tokens_generated, bound);
        assert_eq!(steps.len(), bound);
        assert!(diagnostics.get("stop_reason").is_none());
        // The streamed deltas are the transcript, detokenized incrementally:
        // re-decoding every step would pass bound·(bound+1)/2 tokens.
        let deltas = steps
            .iter()
            .map(|step| step.delta.as_str())
            .collect::<String>();
        assert_eq!(deltas, steps.last().unwrap().text);
        let detokenized = diagnostics["detokenized_tokens"].as_u64().unwrap();
        assert!(
            detokenized <= 4 * bound as u64,
            "detokenized {detokenized} tokens"
        );

        // Each frame emits max_symbols one-token words with zero duration,
        // timed at 80 ms per encoder frame.
        let words = diagnostics["timestamp_words"].as_array().unwrap();
        assert_eq!(words.len(), bound);
        for (index, word) in words.iter().enumerate() {
            let frame = (index / DEFAULT_MAX_SYMBOLS) as f64;
            assert_eq!(word["word"], "a");
            assert!((word["start_time_seconds"].as_f64().unwrap() - frame * 0.08).abs() < 1e-9);
            assert!((word["end_time_seconds"].as_f64().unwrap() - frame * 0.08).abs() < 1e-9);
        }
    }

    #[test]
    fn tdt_words_span_their_tokens_and_absorb_punctuation() {
        let vocab: Vec<String> = ["▁Hel", "lo", "▁world", "▁,", "▁again", "!", "<unk>"]
            .into_iter()
            .map(String::from)
            .collect();
        let decoder = ParakeetDecoder::Vocab(vocab);
        let emissions = [
            (0, 2, 1),
            (1, 3, 2),
            (2, 6, 1),
            (3, 8, 0),
            (6, 9, 0),
            (4, 10, 3),
            (5, 13, 1),
        ];
        let token_ids: Vec<usize> = emissions.iter().map(|(id, ..)| *id).collect();
        let token_frames: Vec<TdtTokenFrame> = emissions
            .iter()
            .map(|(_, frame, duration)| TdtTokenFrame {
                frame: *frame,
                duration: *duration,
            })
            .collect();

        let words = tdt_word_timestamps(
            &token_ids,
            &token_frames,
            |id| decoder.starts_word(id),
            |ids| decoder.decode(ids),
        );

        let spans: Vec<(&str, usize, usize)> = words
            .iter()
            .map(|word| (word.text.as_str(), word.start_frame, word.end_frame))
            .collect();
        // `<unk>` decodes to nothing and continues the word before it.
        assert_eq!(
            spans,
            [("Hello", 2, 5), ("world,", 6, 9), ("again!", 10, 14)]
        );

        let json = decoder.timestamp_words(&token_ids, &token_frames);
        assert_eq!(json[0]["word"], "Hello");
        assert!((json[0]["start_time_seconds"].as_f64().unwrap() - 0.16).abs() < 1e-9);
        assert!((json[0]["end_time_seconds"].as_f64().unwrap() - 0.40).abs() < 1e-9);
        assert!((json[2]["end_time_seconds"].as_f64().unwrap() - 1.12).abs() < 1e-9);
    }

    fn f16_precision() -> DTypeSelection {
        DTypeSelection {
            dtype: DType::F16,
            reason: "test: weight-only F16".into(),
        }
    }

    /// A layer's linear projections in the F32 and F16 tiers (q, k and v
    /// separate).
    fn split_projections(layer: &super::ConformerLayer) -> Vec<&ProjectionLinear> {
        let [q, k, v] = layer.self_attn.qkv.split().expect("separate q, k and v");
        vec![
            &layer.ff1.linear1,
            &layer.ff1.linear2,
            &layer.ff2.linear1,
            &layer.ff2.linear2,
            q,
            k,
            v,
            &layer.self_attn.linear_out,
            &layer.self_attn.linear_pos,
        ]
    }

    /// Every projection the recipe lists is F16 and nothing else is.
    fn assert_projection_split(encoder: &ParakeetEncoder, f16: bool) {
        assert_eq!(encoder.pre_encode.out.as_f16().is_some(), f16);
        for layer in &encoder.layers {
            for projection in split_projections(layer) {
                assert_eq!(projection.as_f16().is_some(), f16);
            }
            assert_eq!(layer.conv.pointwise_conv1.as_f16().is_some(), f16);
            assert_eq!(layer.conv.pointwise_conv2.as_f16().is_some(), f16);
            // The recipe's F32 tensors.
            assert_eq!(layer.conv.depthwise_conv.weight().dtype(), DType::F32);
            assert_eq!(layer.conv.folded_depthwise_weight.dtype(), DType::F32);
            assert_eq!(layer.self_attn.pos_bias_u.dtype(), DType::F32);
        }
        assert_eq!(encoder.pre_encode.conv0.weight().dtype(), DType::F32);
        assert_eq!(encoder.pre_encode.conv3.weight().dtype(), DType::F32);
    }

    fn encode_probe(encoder: &ParakeetEncoder, device: &Device) -> Vec<f32> {
        let frames = 8 * 40 + 1;
        let features = random_log_mel(frames, frames - 1, 17)
            .normalize_and_upload(0..frames, device)
            .unwrap();
        let encoded = whole_utterance_encode(encoder, &features, 40, true);
        host_f32(&encoded).unwrap()
    }

    #[test]
    fn weight_only_f16_encoder_matches_f32_on_cpu() {
        let device = Device::Cpu;
        let reference = encode_probe(&random_encoder(TINY, 31, &device), &device);
        let (encoder, precision) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, 31, &device),
            TINY,
            f16_precision(),
        )
        .unwrap();
        assert!(precision.projections_f16, "{}", precision.reason);
        assert!(precision.self_check.as_ref().unwrap().passed);
        assert_projection_split(&encoder, true);
        let rel = relative_l2(&encode_probe(&encoder, &device), &reference);
        assert!(rel > 0.0 && rel < 1e-2, "relative L2 {rel:e}");
        assert_eq!(
            precision.diagnostics()["projection_weights"],
            serde_json::json!("F16")
        );
    }

    /// Admission's F16 tensor set is exactly the set the loader converts.
    #[test]
    fn f16_projection_names_match_the_loader() {
        let device = Device::Cpu;
        let tensors = random_encoder_tensors(TINY, 37, &device);
        let (encoder, _) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, 37, &device),
            TINY,
            f16_precision(),
        )
        .unwrap();
        let f16_weights = std::iter::once(encoder.pre_encode.out.as_f16().unwrap())
            .chain(encoder.layers.iter().flat_map(|layer| {
                split_projections(layer)
                    .into_iter()
                    .map(|projection| projection.as_f16().unwrap())
                    .chain([
                        layer.conv.pointwise_conv1.as_f16().unwrap(),
                        layer.conv.pointwise_conv2.as_f16().unwrap(),
                    ])
            }))
            .map(|linear| linear.out_features() * linear.in_features())
            .sum::<usize>();
        let named = tensors
            .iter()
            .filter(|(name, _)| parakeet_tensor_is_f16_projection(&format!("encoder.{name}")))
            .map(|(_, tensor)| tensor.elem_count())
            .sum::<usize>();
        assert_eq!(named, f16_weights);
        assert_eq!(
            tensors
                .keys()
                .filter(|name| parakeet_tensor_is_f16_projection(&format!("encoder.{name}")))
                .count(),
            1 + 11 * TINY.layers
        );
        for name in [
            "encoder.pre_encode.out.bias",
            "encoder.pre_encode.conv.3.weight",
            "encoder.layers.0.conv.depthwise_conv.weight",
            "encoder.layers.x.feed_forward1.linear1.weight",
            "joint.enc.weight",
            "decoder.prediction.dec_rnn.lstm.weight_ih_l0",
        ] {
            assert!(!parakeet_tensor_is_f16_projection(name), "{name}");
        }
    }

    #[test]
    fn f32_precision_keeps_every_weight_dense() {
        let device = Device::Cpu;
        let (encoder, precision) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, 32, &device),
            TINY,
            DTypeSelection {
                dtype: DType::F32,
                reason: "test: F32".into(),
            },
        )
        .unwrap();
        assert!(!precision.projections_f16);
        assert!(precision.self_check.is_none());
        assert_projection_split(&encoder, false);
    }

    fn bf16_precision() -> DTypeSelection {
        DTypeSelection {
            dtype: DType::BF16,
            reason: "test: BF16 projections".into(),
        }
    }

    fn bf16_encoder(seed: u64, device: &Device) -> ParakeetEncoder {
        let (encoder, precision) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, seed, device),
            TINY,
            bf16_precision(),
        )
        .unwrap();
        assert!(!precision.projections_f16);
        assert!(precision.self_check.is_none());
        assert_eq!(encoder.projection_dtype, DType::BF16);
        encoder
    }

    /// The BF16 tensors of a BF16-tier encoder, by role. Everything else it
    /// holds stays F32.
    fn bf16_tensors(encoder: &ParakeetEncoder) -> Vec<Tensor> {
        fn dense(projection: &ProjectionLinear) -> &Linear {
            match projection {
                ProjectionLinear::Dense(linear) => linear,
                ProjectionLinear::F16(_) => panic!("weight-only F16 projection in the BF16 tier"),
            }
        }
        let linear_tensors = |linear: &Linear| {
            std::iter::once(linear.weight().clone()).chain(linear.bias().cloned())
        };
        let mut tensors = linear_tensors(dense(&encoder.pre_encode.out)).collect::<Vec<_>>();
        for layer in &encoder.layers {
            let QkvProjection::Fused(qkv) = &layer.self_attn.qkv else {
                panic!("separate q, k and v in the BF16 tier");
            };
            assert_eq!(qkv.weight().dims(), [3 * TINY.dim, TINY.dim]);
            for linear in [
                dense(&layer.ff1.linear1),
                dense(&layer.ff1.linear2),
                dense(&layer.ff2.linear1),
                dense(&layer.ff2.linear2),
                qkv,
                dense(&layer.self_attn.linear_out),
                dense(&layer.self_attn.linear_pos),
            ] {
                tensors.extend(linear_tensors(linear));
            }
            for pointwise in [&layer.conv.pointwise_conv1, &layer.conv.pointwise_conv2] {
                let ConvPointwise::Bf16(linear) = pointwise else {
                    panic!("pointwise conv outside the BF16 tier");
                };
                tensors.extend(linear_tensors(linear));
            }
        }
        tensors
    }

    /// The BF16 tier converts exactly the tensors admission prices at BF16
    /// (`parakeet_tensor_is_bf16_projection`), and keeps the recipe's F32
    /// tensors F32.
    #[test]
    fn bf16_projection_names_match_the_loader() {
        let device = Device::Cpu;
        let tensors = random_encoder_tensors(TINY, 38, &device);
        let encoder = bf16_encoder(38, &device);
        let converted = bf16_tensors(&encoder);
        assert!(converted.iter().all(|tensor| tensor.dtype() == DType::BF16));
        let named = |filter: fn(&str) -> bool| {
            tensors
                .iter()
                .filter(|(name, _)| filter(&format!("encoder.{name}")))
                .map(|(_, tensor)| tensor.elem_count())
                .sum::<usize>()
        };
        assert_eq!(
            converted.iter().map(Tensor::elem_count).sum::<usize>(),
            named(parakeet_tensor_is_bf16_projection)
        );
        assert_eq!(
            named(parakeet_tensor_is_bf16_projection) - named(parakeet_tensor_is_f16_projection),
            TINY.dim
        );
        assert!(parakeet_tensor_is_bf16_projection(
            "encoder.pre_encode.out.bias"
        ));
        for name in [
            "encoder.pre_encode.conv.3.weight",
            "encoder.layers.0.conv.depthwise_conv.weight",
            "encoder.layers.0.self_attn.pos_bias_u",
            "encoder.layers.0.norm_out.weight",
            "joint.enc.weight",
            "decoder.prediction.dec_rnn.lstm.weight_ih_l0",
        ] {
            assert!(!parakeet_tensor_is_bf16_projection(name), "{name}");
        }
        assert!(encoder.pre_encode.out.as_f16().is_none());
        assert_eq!(encoder.pre_encode.conv0.weight().dtype(), DType::F32);
        assert_eq!(encoder.pre_encode.conv6.weight().dtype(), DType::F32);
        for layer in &encoder.layers {
            assert_eq!(layer.conv.depthwise_conv.weight().dtype(), DType::F32);
            assert_eq!(layer.conv.folded_depthwise_weight.dtype(), DType::F32);
            assert_eq!(layer.conv.folded_depthwise_bias.dtype(), DType::F32);
            assert_eq!(layer.self_attn.pos_bias_u.dtype(), DType::F32);
            assert_eq!(layer.self_attn.pos_bias_v.dtype(), DType::F32);
            assert_eq!(layer.norm_ff1.weight().dtype(), DType::F32);
            assert_eq!(layer.norm_out.weight().dtype(), DType::F32);
        }
    }

    /// A model over `encoder` whose joint reads the encoder output through
    /// random weights (the zero predictor adds nothing), so its TDT decode
    /// decisions follow the encoder.
    fn encoder_driven_model(device: &Device, encoder: ParakeetEncoder) -> ParakeetAsrModel {
        let mut rng = XorShift(0x7D7_0001);
        let mut random = |shape: &[usize], scale: f32| {
            let values = rng.values(shape.iter().product(), scale, 0.0);
            Tensor::from_vec(values, shape, device).unwrap()
        };
        let enc = random(&[PRED_HIDDEN, TINY.dim], 2.0 / (TINY.dim as f32).sqrt());
        let out = random(&[5, PRED_HIDDEN], 4.0 / (PRED_HIDDEN as f32).sqrt());
        let bias = random(&[5], 0.5);
        let mut model = forced_label_model(device);
        model.network.encoder = encoder;
        model.network.joint.enc = Linear::new(enc, None);
        model.network.joint.out = Linear::new(out, Some(bias));
        model
    }

    /// The joint's `[T, labels + durations]` logits over `encoded` with a
    /// zero predictor output.
    fn joint_logits(model: &ParakeetAsrModel, encoded: &Tensor) -> Tensor {
        let predictor = Tensor::zeros((1, 1, PRED_HIDDEN), DType::F32, encoded.device()).unwrap();
        model
            .network
            .joint
            .joint_after_projection(encoded, &predictor)
            .unwrap()
            .squeeze(2)
            .unwrap()
            .squeeze(0)
            .unwrap()
    }

    /// Sets the joint output bias of `models` to minus the mean logit of the
    /// first over the frames of `encoded`, so a frame's decisions turn on
    /// what sets it apart from the others (frames of one input share most of
    /// their encoding).
    fn center_joint(models: [&mut ParakeetAsrModel; 2], encoded: &Tensor) {
        let without_bias = {
            let out = &models[0].network.joint.out;
            let bias = out.bias().unwrap();
            joint_logits(models[0], encoded)
                .broadcast_sub(bias)
                .unwrap()
        };
        let bias = without_bias.mean(0).unwrap().neg().unwrap();
        for model in models {
            let weight = model.network.joint.out.weight().clone();
            model.network.joint.out = Linear::new(weight, Some(bias.clone()));
        }
    }

    /// Each frame's TDT decisions (label, duration) by the joint over
    /// `encoded` with a zero predictor output, with the gap between each
    /// winning logit and its runner-up.
    fn frame_decisions(
        model: &ParakeetAsrModel,
        encoded: &Tensor,
    ) -> Vec<((usize, usize), [f32; 2])> {
        let classes = model.network.joint.num_classes_with_blank;
        let margin = |values: &[f32]| {
            let mut sorted = values.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            sorted[0] - sorted[1]
        };
        joint_logits(model, encoded)
            .to_vec2::<f32>()
            .unwrap()
            .into_iter()
            .map(|row| {
                let frame = Tensor::new(row.as_slice(), encoded.device()).unwrap();
                (
                    tdt_label_and_duration(&frame, classes).unwrap(),
                    [margin(&row[..classes]), margin(&row[classes..])],
                )
            })
            .collect()
    }

    /// The BF16 tier (BF16 projections and GEMMs, F32 residual stream,
    /// norms, depthwise conv and attention core) stays within BF16 rounding
    /// of the F32 encoder and keeps the TDT decisions of a tiny decode: a
    /// confident joint decodes identically, and a joint centred on the
    /// input changes no decision whose F32 margin exceeds twice the largest
    /// logit change BF16 makes (a decision inside that band is a tie to
    /// within BF16 rounding).
    #[test]
    fn bf16_encoder_matches_f32_and_keeps_decode_decisions() {
        let device = Device::Cpu;
        let f32_encoder = random_encoder(TINY, 31, &device);
        let bf16 = bf16_encoder(31, &device);
        let reference = encode_probe(&f32_encoder, &device);
        let actual = encode_probe(&bf16, &device);
        let rel = relative_l2(&actual, &reference);
        let max_abs = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let scale = reference
            .iter()
            .fold(0f32, |max, value| max.max(value.abs()));
        eprintln!(
            "Parakeet BF16 vs F32 encoder (TINY, CPU): relative L2 {rel:.3e}, max |diff| {max_abs:.3e} of max |x| {scale:.3}"
        );
        assert!(
            rel > 0.0 && rel < BF16_ENCODE_RELATIVE_L2,
            "relative L2 {rel:e}"
        );

        let mut f32_model = encoder_driven_model(&device, f32_encoder);
        let mut bf16_model = encoder_driven_model(&device, bf16);
        // 4 s of audio, 50 encoder frames: a tone that changes pitch and
        // level every quarter second over faint noise.
        let noise = XorShift(13).values(64_037, 0.01, 0.0);
        let audio = noise
            .iter()
            .enumerate()
            .map(|(n, noise)| {
                let segment = n / 4_000;
                let pitch = 200.0 + 600.0 * (segment % 7) as f32;
                let level = 0.05 + 0.1 * (segment % 3) as f32;
                let phase = 2.0 * std::f32::consts::PI * pitch * n as f32 / 16_000.0;
                level * phase.sin() + noise
            })
            .collect::<Vec<_>>();
        let artifacts = [&f32_model, &bf16_model].map(|model| {
            model
                .prepare_retained_encoder(&audio, 16_000, Some("en"), &ready)
                .unwrap()
        });
        assert_eq!(artifacts[1].encoded.dtype(), DType::F32);
        let compare = |f32_model: &ParakeetAsrModel, bf16_model: &ParakeetAsrModel| {
            let perturbation = max_abs_diff(
                &joint_logits(f32_model, &artifacts[0].encoded),
                &joint_logits(bf16_model, &artifacts[1].encoded),
            );
            let f32_decisions = frame_decisions(f32_model, &artifacts[0].encoded);
            let bf16_decisions = frame_decisions(bf16_model, &artifacts[1].encoded);
            assert_eq!(f32_decisions.len(), 50);
            (f32_decisions, bf16_decisions, perturbation)
        };

        // A confident joint: every decision clears the band, so the whole
        // retained decode (labels, durations, timestamps) is identical.
        let (f32_decisions, bf16_decisions, perturbation) = compare(&f32_model, &bf16_model);
        let smallest = f32_decisions
            .iter()
            .flat_map(|(_, margins)| *margins)
            .fold(f32::INFINITY, f32::min);
        assert!(
            smallest > 2.0 * perturbation,
            "margin {smallest} vs logit change {perturbation}"
        );
        assert_eq!(
            bf16_decisions.iter().map(|d| d.0).collect::<Vec<_>>(),
            f32_decisions.iter().map(|d| d.0).collect::<Vec<_>>()
        );
        let (f32_steps, f32_diagnostics) = decode_until_finished(&f32_model, &artifacts[0], 500);
        let (bf16_steps, bf16_diagnostics) = decode_until_finished(&bf16_model, &artifacts[1], 500);
        assert_eq!(
            bf16_steps.last().unwrap().text,
            f32_steps.last().unwrap().text
        );
        for key in [
            "tdt_joint_steps",
            "tdt_emitted_tokens",
            "tdt_blank_steps",
            "timestamp_words",
        ] {
            assert_eq!(bf16_diagnostics[key], f32_diagnostics[key], "{key}");
        }
        eprintln!(
            "Parakeet BF16 vs F32 confident decode: identical over {} joint steps ({} tokens), smallest margin {smallest:.3e}, largest logit change {perturbation:.3e}",
            f32_diagnostics["tdt_joint_steps"], f32_diagnostics["tdt_emitted_tokens"]
        );

        // A centred joint: decisions vary frame to frame.
        center_joint([&mut f32_model, &mut bf16_model], &artifacts[0].encoded);
        let (f32_decisions, bf16_decisions, perturbation) = compare(&f32_model, &bf16_model);
        let band = 2.0 * perturbation;
        let (mut decisive, mut ties, mut flipped) = (0, 0, 0);
        for (frame, ((f32_decision, margins), (bf16_decision, _))) in
            f32_decisions.iter().zip(&bf16_decisions).enumerate()
        {
            let heads = [
                (f32_decision.0, bf16_decision.0),
                (f32_decision.1, bf16_decision.1),
            ];
            for ((f32_choice, bf16_choice), margin) in heads.into_iter().zip(margins) {
                if *margin > band {
                    decisive += 1;
                    assert_eq!(bf16_choice, f32_choice, "frame {frame}: margin {margin}");
                } else {
                    ties += 1;
                    flipped += usize::from(bf16_choice != f32_choice);
                }
            }
        }
        eprintln!(
            "Parakeet BF16 vs F32 centred joint: {decisive} decisive decisions identical; {flipped} of {ties} within {band:.3e} of a tie flipped"
        );
        let labels = f32_decisions.iter().map(|d| d.0 .0).collect::<Vec<_>>();
        let durations = f32_decisions.iter().map(|d| d.0 .1).collect::<Vec<_>>();
        assert!(labels.iter().any(|label| *label != labels[0]));
        assert!(durations.iter().any(|duration| *duration != durations[0]));
        // Centring leaves small logits: 69 of 100 decisions clear the band
        // here, and none of those changes.
        assert!(decisive >= 60, "{decisive} of 100 decisions decisive");
        assert_eq!(
            bf16_model.weight_precision_diagnostics()["projection_weights"],
            serde_json::json!("BF16")
        );
    }

    /// A padded BF16 batch encodes each row as the row alone does in BF16,
    /// within BF16 rounding: the CPU product regroups its F32 sums by batch
    /// shape before narrowing to BF16, so one rounding step can differ.
    #[test]
    fn bf16_batched_encode_matches_solo_bf16_encode() {
        let device = Device::Cpu;
        let mut model = forced_label_model(&device);
        model.network.encoder = bf16_encoder(1, &device);
        let lengths = [1usize, 3, 37, 64, 125];
        let features = lengths
            .iter()
            .zip(31..)
            .map(|(len, seed)| features_of_frames(&model, *len, seed))
            .collect::<Vec<_>>();
        let solo = features
            .iter()
            .map(|features| model.encode_host_features(features, None, &ready).unwrap())
            .collect::<Vec<_>>();
        let rows = features
            .iter()
            .map(|features| batch_row(features, &ready))
            .collect::<Vec<_>>();
        let batched = model.prepare_retained_encoder_batch(&rows).unwrap();
        let mut worst = 0f64;
        for ((artifact, solo), len) in batched.into_iter().zip(&solo).zip(lengths) {
            let artifact = artifact.unwrap();
            assert_eq!(artifact.encoded.dtype(), DType::F32);
            assert_eq!(artifact.encoded.dims(), solo.encoded.dims());
            assert_eq!(
                artifact.resident_tensor_bytes().unwrap(),
                (len * TINY.dim * 4) as u64
            );
            let rel = relative_l2(
                &host_f32(&artifact.encoded).unwrap(),
                &host_f32(&solo.encoded).unwrap(),
            );
            worst = worst.max(rel);
        }
        eprintln!("Parakeet BF16 batched vs solo encode on CPU: worst relative L2 {worst:.3e}");
        assert!(worst < BF16_ENCODE_RELATIVE_L2, "relative L2 {worst:e}");
    }

    /// `IZWI_PARAKEET_ASR_DTYPE` on CUDA: BF16 by default and for `bf16`,
    /// F32 for `f32` (today's path), anything else ignored visibly; a device
    /// without BF16 stays F32. CPU stays F32 and Metal keeps weight-only F16.
    #[test]
    fn cuda_projection_precision_defaults_to_bf16_with_an_f32_kill_switch() {
        let select = parakeet_cuda_projection_precision;
        assert_eq!(select(true, None).dtype, DType::BF16);
        assert_eq!(select(true, Some("  ")).dtype, DType::BF16);
        assert_eq!(select(true, Some("bf16")).dtype, DType::BF16);
        let pinned = select(true, Some("f32"));
        assert_eq!(pinned.dtype, DType::F32);
        assert!(
            pinned.reason.contains("IZWI_PARAKEET_ASR_DTYPE=f32"),
            "{}",
            pinned.reason
        );
        let ignored = select(true, Some("f16"));
        assert_eq!(ignored.dtype, DType::BF16);
        assert!(ignored.reason.contains("ignored"), "{}", ignored.reason);
        assert_eq!(select(false, None).dtype, DType::F32);

        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let profile = |kind| crate::backends::DeviceProfile {
            device: Device::Cpu,
            kind,
            capabilities: crate::backends::DeviceCapabilities {
                supports_bf16: kind == crate::backends::DeviceKind::Cuda,
                supports_f16: kind != crate::backends::DeviceKind::Cpu,
                ..Default::default()
            },
            memory_pool: None,
        };
        use crate::backends::DeviceKind::{Cpu, Cuda, Metal};
        let resolve = |kind| resolve_parakeet_projection_precision(&profile(kind)).dtype;
        std::env::remove_var(PARAKEET_ASR_DTYPE_ENV);
        assert_eq!(
            [resolve(Cuda), resolve(Metal), resolve(Cpu)],
            [DType::BF16, DType::F16, DType::F32]
        );
        std::env::set_var(PARAKEET_ASR_DTYPE_ENV, "f32");
        let pinned = [resolve(Cuda), resolve(Metal), resolve(Cpu)];
        std::env::set_var(PARAKEET_ASR_DTYPE_ENV, "bf16");
        let bf16 = [resolve(Cuda), resolve(Cpu)];
        std::env::remove_var(PARAKEET_ASR_DTYPE_ENV);
        assert_eq!(pinned, [DType::F32; 3]);
        assert_eq!(bf16, [DType::BF16, DType::F32]);
    }

    /// A weight F16 cannot represent turns the F16 encoder non-finite; the
    /// self-check must catch it and reload in F32 with the reason recorded.
    #[test]
    fn failed_self_check_falls_back_to_f32() {
        let device = Device::Cpu;
        let mut tensors = random_encoder_tensors(TINY, 33, &device);
        let name = "layers.1.feed_forward2.linear2.weight";
        let weight = &tensors[name];
        let mut values = host_f32(weight).unwrap();
        values[0] = 1.0e5;
        let replaced = Tensor::from_vec(values, weight.dims(), &device).unwrap();
        tensors.insert(name.into(), replaced);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);

        let (encoder, precision) =
            ParakeetEncoder::load_with_precision(vb, TINY, f16_precision()).unwrap();
        assert!(!precision.projections_f16);
        let report = precision.self_check.as_ref().unwrap();
        assert!(!report.passed);
        assert!(
            precision.reason.contains("fell back to F32"),
            "{}",
            precision.reason
        );
        assert!(
            precision.reason.contains("layers.1"),
            "{}",
            precision.reason
        );
        assert_projection_split(&encoder, false);
        assert!(encode_probe(&encoder, &device)
            .iter()
            .all(|value| value.is_finite()));
    }

    /// A bad conversion of the probed projection fails the transform check
    /// even when the encoder output stays finite.
    #[test]
    fn self_check_compares_the_conversion_with_the_original_weights() {
        let device = Device::Cpu;
        let (mut encoder, _) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, 34, &device),
            TINY,
            f16_precision(),
        )
        .unwrap();
        let other = ParakeetEncoder::load(
            random_encoder_vb(TINY, 35, &device),
            TINY,
            Projections::F16(&device),
        )
        .unwrap();
        encoder.layers[0].ff1.linear1 = other.layers[0].ff1.linear1.clone();
        let report = encoder.weight_only_f16_self_check(&random_encoder_vb(TINY, 34, &device));
        assert!(!report.passed);
        assert!(
            report.reason.contains("feed_forward1.linear1"),
            "{}",
            report.reason
        );
    }

    /// Production loading on Metal: the projections take ggml's
    /// f16 x f32 kernel, the self-check passes on the device, and the
    /// encoder matches the CPU F32 encoder.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_weight_only_f16_encoder_engages_ggml_and_matches_cpu_f32() {
        use crate::kernels::f16_weight_linear::F16WeightLinearPath;

        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_weight_only_f16_encoder_engages_ggml_and_matches_cpu_f32",
        ) else {
            return;
        };
        let reference = encode_probe(&random_encoder(TINY, 36, &Device::Cpu), &Device::Cpu);
        let (encoder, precision) = ParakeetEncoder::load_with_precision(
            random_encoder_vb(TINY, 36, &device),
            TINY,
            f16_precision(),
        )
        .unwrap();
        assert!(precision.projections_f16, "{}", precision.reason);
        assert_projection_split(&encoder, true);
        for layer in &encoder.layers {
            for projection in [&layer.ff1.linear1, &layer.self_attn.linear_pos] {
                assert_eq!(
                    projection.as_f16().unwrap().path(),
                    F16WeightLinearPath::GgmlF16F32
                );
            }
            assert_eq!(
                layer.conv.pointwise_conv2.as_f16().unwrap().path(),
                F16WeightLinearPath::GgmlF16F32
            );
        }
        let rel = relative_l2(&encode_probe(&encoder, &device), &reference);
        assert!(rel < 1e-2, "relative L2 {rel:e}");
    }

    #[test]
    fn max_abs_diff_treats_nan_as_infinitely_far() {
        let device = Device::Cpu;
        let finite = Tensor::new(&[1f32, 2.0], &device).unwrap();
        let with_nan = Tensor::new(&[1f32, f32::NAN], &device).unwrap();
        assert_eq!(max_abs_diff(&finite, &with_nan), f32::INFINITY);
        assert_eq!(max_abs_diff(&finite, &finite), 0.0);
    }

    #[test]
    fn encoder_window_plan_partitions_frames_with_context() {
        let w = EncoderWindowing::V3;
        let plan = |encoded_len| {
            plan_encoder_windows(encoded_len, w.frames, w.left_context, w.right_context).unwrap()
        };
        for encoded_len in [1, 895, 896, 1023, 1024, 1025, 1793, 45_000] {
            let windows = plan(encoded_len);
            let mut kept_until = 0;
            for window in &windows {
                assert_eq!(window.keep_start, kept_until, "E={encoded_len}");
                assert!(window.start <= window.keep_start && window.keep_start < window.keep_end);
                assert!(window.keep_end <= window.end && window.end <= encoded_len);
                kept_until = window.keep_end;
                if encoded_len > w.frames {
                    assert_eq!(window.end - window.start, w.frames, "E={encoded_len}");
                }
                assert!(window.start == 0 || window.keep_start - window.start >= w.left_context);
                assert!(
                    window.end == encoded_len || window.end - window.keep_end >= w.right_context
                );
                // Both STFT tails: valid % 8 != 0 (one padding column past
                // 8E - 3) and valid % 8 == 0 (padding column at 8E).
                for feature_frames in [8 * encoded_len - 2, 8 * encoded_len + 1] {
                    let columns = window.feature_columns(feature_frames);
                    assert_eq!(columns.start, 8 * window.start);
                    assert_eq!(columns.end, (8 * window.end).min(feature_frames));
                    assert_eq!(subsampled_len_3x(columns.len()), window.end - window.start);
                }
            }
            assert_eq!(kept_until, encoded_len, "E={encoded_len}");
        }

        let whole = |encoded_len| EncoderWindow {
            start: 0,
            end: encoded_len,
            keep_start: 0,
            keep_end: encoded_len,
        };
        assert_eq!(plan(1), vec![whole(1)]);
        assert_eq!(plan(1024), vec![whole(1024)]);
        assert_eq!(
            plan(1025),
            vec![
                EncoderWindow {
                    start: 0,
                    end: 1024,
                    keep_start: 0,
                    keep_end: 896,
                },
                EncoderWindow {
                    start: 1,
                    end: 1025,
                    keep_start: 896,
                    keep_end: 1025,
                },
            ]
        );
        assert_eq!(plan(1793).len(), 3);
        // 60 minutes: one 896-frame head, 57 centres of 768 and a tail.
        assert_eq!(plan(45_000).len(), 59);
        assert!(plan(0).is_empty());
        assert!(plan_encoder_windows(2048, 256, 128, 128).is_err());
    }

    #[test]
    fn single_window_encode_matches_whole_utterance_encode() {
        let device = Device::Cpu;
        let encoder = random_encoder(TINY, 3, &device);
        // 203 % 8 != 0: the padding column shares the last encoder frame.
        // 200 % 8 == 0: it would add a frame, which attention no longer sees.
        for valid in [203usize, 200] {
            let frames = valid + 1;
            let log_mel = random_log_mel(frames, valid, valid as u64);
            let encoded_len = subsampled_len_3x(valid);
            let windowed = encode_windows(
                &encoder,
                frames,
                encoded_len,
                EncoderWindowing::V3,
                &mut |columns| log_mel.normalize_and_upload(columns, &device),
                &|| Ok(()),
            )
            .unwrap();
            assert_eq!(windowed.windows, 1);
            assert_eq!(windowed.encoded.dims(), &[1, encoded_len, TINY.dim]);

            let features = log_mel.normalize_and_upload(0..frames, &device).unwrap();
            let before = whole_utterance_encode(&encoder, &features, encoded_len, false);
            if valid % 8 != 0 {
                assert_eq!(
                    windowed.encoded.to_vec3::<f32>().unwrap(),
                    before.to_vec3::<f32>().unwrap()
                );
            } else {
                let truncated = whole_utterance_encode(&encoder, &features, encoded_len, true);
                assert!(max_abs_diff(&windowed.encoded, &truncated) <= 1e-5);
                assert!(max_abs_diff(&windowed.encoded, &before) > 1e-4);
            }
        }
    }

    #[test]
    fn windowed_encode_with_neutral_attention_matches_whole_utterance_encode() {
        let device = Device::Cpu;
        let mut encoder = random_encoder(TINY, 5, &device);
        // With attention silenced only per-frame blocks and kernel-9
        // convolutions remain, so context of 4 frames per layer, plus the
        // first subsampled frame, makes every kept frame exact.
        for layer in &mut encoder.layers {
            let weight = Tensor::zeros((TINY.dim, TINY.dim), DType::F32, &device).unwrap();
            layer.self_attn.linear_out = Linear::new(weight, None).into();
        }
        let valid = 8 * 150 + 5;
        let frames = valid + 1;
        let encoded_len = subsampled_len_3x(valid);
        // One normalization for every window, so only stitching can differ.
        let features = random_log_mel(frames, valid, 21)
            .normalize_and_upload(0..frames, &device)
            .unwrap();
        let reference = whole_utterance_encode(&encoder, &features, encoded_len, true);
        let windowed = |context: usize| {
            let windowing = EncoderWindowing {
                frames: 48,
                left_context: context,
                right_context: context,
            };
            encode_windows(
                &encoder,
                frames,
                encoded_len,
                windowing,
                &mut |columns| {
                    Ok(features
                        .narrow(2, columns.start, columns.len())?
                        .contiguous()?)
                },
                &|| Ok(()),
            )
            .unwrap()
        };

        let exact = windowed(4 * TINY.layers + 2);
        assert!(exact.windows > 3);
        assert!(max_abs_diff(&exact.encoded, &reference) <= 1e-5);
        // Too little context leaves window-edge effects in kept frames.
        assert!(max_abs_diff(&windowed(2).encoded, &reference) > 1e-3);
    }

    /// A long input's windows stacked as the rows of one batch encode as the
    /// sequential windows do (within the batched-encode bound), and the
    /// stacked encode checks for cancellation before every batch.
    #[test]
    fn stacked_windows_match_sequential_windows() {
        let device = Device::Cpu;
        let encoder = random_encoder(TINY, 11, &device);
        let valid = 8 * 120 + 3;
        let frames = valid + 1;
        let encoded_len = subsampled_len_3x(valid);
        let log_mel = random_log_mel(frames, valid, 33);
        let windowing = EncoderWindowing {
            frames: 32,
            left_context: 8,
            right_context: 8,
        };
        let run = |rows: usize, checks: &Cell<usize>| {
            encode_windows_stacked(
                &encoder,
                frames,
                encoded_len,
                windowing,
                rows,
                &mut |columns| log_mel.normalize_and_upload(columns, &device),
                &|| {
                    checks.set(checks.get() + 1);
                    Ok(())
                },
            )
            .unwrap()
        };
        let sequential_checks = Cell::new(0);
        let sequential = run(1, &sequential_checks);
        assert!(sequential.windows >= 5, "{} windows", sequential.windows);
        assert_eq!(sequential_checks.get(), sequential.windows);
        let scale = sequential
            .encoded
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        for rows in [2, 3, sequential.windows + 4] {
            let checks = Cell::new(0);
            let stacked = run(rows, &checks);
            assert_eq!(stacked.windows, sequential.windows);
            assert_eq!(stacked.encoded.dims(), sequential.encoded.dims());
            assert_eq!(checks.get(), sequential.windows.div_ceil(rows), "{rows} rows");
            let diff = max_abs_diff(&stacked.encoded, &sequential.encoded);
            assert!(
                diff <= BATCHED_ENCODE_TOLERANCE * scale.max(1.0),
                "{rows} rows: max |diff| {diff} of {scale}"
            );
        }
    }

    #[test]
    fn windowed_encode_stops_at_a_cancelled_checkpoint() {
        let device = Device::Cpu;
        let encoder = random_encoder(TINY, 9, &device);
        let valid = 8 * 100;
        let log_mel = random_log_mel(valid + 1, valid, 4);
        let windowing = EncoderWindowing {
            frames: 32,
            left_context: 8,
            right_context: 8,
        };
        let encoded_windows = Cell::new(0);
        let cancelled = Cell::new(false);

        let result = encode_windows(
            &encoder,
            valid + 1,
            subsampled_len_3x(valid),
            windowing,
            &mut |columns| {
                encoded_windows.set(encoded_windows.get() + 1);
                // The request is cancelled while its first window encodes.
                cancelled.set(true);
                log_mel.normalize_and_upload(columns, &device)
            },
            &|| {
                if cancelled.get() {
                    return Err(Error::Cancelled("parakeet-test".into()));
                }
                Ok(())
            },
        );

        assert!(matches!(result, Err(Error::Cancelled(_))));
        assert_eq!(encoded_windows.get(), 1);
    }

    #[test]
    fn retained_encoder_artifact_reports_its_windows() {
        let device = Device::Cpu;
        let mut model = forced_label_model(&device);
        model.encoder_windowing = EncoderWindowing {
            frames: 32,
            left_context: 10,
            right_context: 10,
        };
        // 4 s: 400 valid feature frames, 50 encoder frames, three windows.
        let audio = XorShift(13).values(64_037, 0.1, 0.0);

        let artifact = model
            .prepare_retained_encoder(&audio, 16_000, Some("en"), &|| Ok(()))
            .unwrap();
        assert_eq!(artifact.encoded_len, 50);
        assert_eq!(artifact.encoder_windows, 3);
        assert_eq!(artifact.encoded.dims(), &[1, 50, TINY.dim]);
        let state = model.start_retained_decode().unwrap();
        let diagnostics = model
            .retained_decode_output(&artifact, &state)
            .diagnostics
            .unwrap();
        assert_eq!(diagnostics["encoded_frames"], 50);
        assert_eq!(diagnostics["encoder_windows"], 3);
        assert_eq!(diagnostics["encoder_window_frames"], 32);
        assert_eq!(diagnostics["encoder_window_left_context"], 10);
        assert_eq!(diagnostics["encoder_window_right_context"], 10);
        assert!(diagnostics["encode_ms"].as_f64().is_some());

        let checkpoints = Cell::new(0);
        let cancelled = model.prepare_retained_encoder(&audio, 16_000, None, &|| {
            checkpoints.set(checkpoints.get() + 1);
            if checkpoints.get() > 1 {
                return Err(Error::Cancelled("parakeet-test".into()));
            }
            Ok(())
        });
        assert!(matches!(cancelled, Err(Error::Cancelled(_))));
        assert_eq!(checkpoints.get(), 2);
    }

    #[test]
    fn preparation_pricing_is_bounded_by_the_window() {
        // Pricing and the encoder read `IZWI_PARAKEET_ENCODER_GRAPHS`.
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        const GIB: u64 = 1 << 30;
        let minute = 60 * 16_000;
        let total = |samples: usize, backend: BackendKind| {
            let bytes = parakeet_preparation_bytes(samples, backend).unwrap();
            bytes.host_bytes + bytes.device_bytes().unwrap()
        };
        let backends = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

        for backend in backends {
            let mut previous = 0;
            let lengths = (0..=60 * minute)
                .step_by(16_000 * 7 + 13)
                .chain(1_300_000..1_320_000)
                .chain([60 * minute]);
            let mut lengths = lengths.collect::<Vec<_>>();
            lengths.sort_unstable();
            for samples in lengths {
                let bytes = total(samples, backend);
                assert!(bytes >= previous, "{backend:?} pricing fell at {samples}");
                previous = bytes;
            }
            // CUDA stacks four windows of a long input (about 1.4 GiB).
            let bound = if backend == BackendKind::Cuda { 3 * GIB } else { 5 * GIB / 2 };
            assert!(total(60 * minute, backend) <= bound, "{backend:?}");
        }

        // E = 1024 frames: the window term is final from here on. It was
        // report 11's 832 MiB before the stem ran by slices (its first
        // convolution held 512 MiB of it); one 128-frame stem slice holds
        // three 32 MiB planes, which Metal's pool does not round.
        let full_window = |backend| {
            parakeet_preparation_bytes(8192 * 160, backend)
                .unwrap()
                .window_bytes
        };
        assert_eq!(full_window(BackendKind::Cpu), (832 - 512 + 96) << 20);
        assert_eq!(full_window(BackendKind::Metal), (832 - 512 + 96) << 20);
        // A longer input stacks its windows as the rows of one batch on
        // CUDA: two windows at 2 minutes, then four; the CPU and Metal run
        // them one at a time, at the one-window price.
        assert_eq!(backends.map(window_batch_rows), [1, 1, 4]);
        for minutes in [2, 10, 30, 60] {
            for backend in backends {
                let bytes = parakeet_preparation_bytes(minutes * minute, backend).unwrap();
                let rows = if minutes == 2 { 2 } else { 4 }.min(window_batch_rows(backend));
                assert_eq!(
                    bytes.window_bytes,
                    encoder_window_row_bytes(1024, rows, 1, backend).unwrap(),
                    "{backend:?} at {minutes} minutes"
                );
            }
            assert_eq!(
                parakeet_preparation_bytes(minutes * minute, BackendKind::Cpu)
                    .unwrap()
                    .window_bytes,
                full_window(BackendKind::Cpu)
            );
        }
        for samples in [16_000, 7 * minute + 4_321, 60 * minute] {
            assert!(total(samples, BackendKind::Metal) >= total(samples, BackendKind::Cpu));
        }
    }

    /// The host worker computes the log-mel; the device job then holds one
    /// window's normalized copy instead of the whole log-mel pipeline.
    #[test]
    fn host_features_price_the_log_mel_and_one_window_upload() {
        // Pricing and the encoder read `IZWI_PARAKEET_ENCODER_GRAPHS`.
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let device = Device::Cpu;
        let model = forced_label_model(&device);
        // 3 s at 8 kHz: 300 valid feature frames, 38 encoder frames.
        let audio = XorShift(17).values(24_000, 0.1, 0.0);
        let features = model.prepare_host_features(&audio, 8_000).unwrap();
        assert_eq!(features.encoded_len(), 38);
        assert_eq!(features.feature_frames(), 301);
        assert_eq!(features.input_samples(), 24_000);
        assert!(features.single_window());
        assert_eq!(features.window_frames(), 38);
        assert!(features.host_bytes().unwrap() >= 128 * 301 * 4);

        let artifact = model
            .encode_host_features(&features, Some("en"), &|| Ok(()))
            .unwrap();
        let reference = model
            .prepare_retained_encoder(&audio, 8_000, Some("en"), &|| Ok(()))
            .unwrap();
        assert_eq!(
            artifact.encoded.to_vec3::<f32>().unwrap(),
            reference.encoded.to_vec3::<f32>().unwrap()
        );
        assert_eq!(artifact.feature_frames, 300);
        assert_eq!(artifact.resampled_samples, 48_000);

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let solo = parakeet_preparation_bytes(48_000, backend).unwrap();
            let device_job = parakeet_encode_row_bytes(&features, 38, 1, backend).unwrap();
            assert_eq!(device_job.host_bytes, 128 * 301 * 4);
            assert!(device_job.window_bytes <= solo.window_bytes);
            assert_eq!(device_job.output_bytes, solo.output_bytes);
        }
        // An empty or rate-less input never reaches the device.
        assert!(model.prepare_host_features(&[], 16_000).is_err());
        assert!(model.prepare_host_features(&audio, 0).is_err());
    }

    /// Host features of a 16 kHz input that encodes to `encoded_len` frames;
    /// its STFT padding column shares the last encoder frame.
    fn features_of_frames(
        model: &ParakeetAsrModel,
        encoded_len: usize,
        seed: u64,
    ) -> ParakeetHostFeatures {
        let samples = (8 * encoded_len - 3) * 160 + 77;
        let audio = XorShift(seed).values(samples, 0.1, 0.0);
        let features = model.prepare_host_features(&audio, 16_000).unwrap();
        assert_eq!(features.encoded_len(), encoded_len);
        features
    }

    fn batch_row<'a>(
        features: &'a ParakeetHostFeatures,
        checkpoint: &'a dyn Fn() -> Result<()>,
    ) -> ParakeetEncoderBatchRow<'a> {
        ParakeetEncoderBatchRow {
            features,
            language: Some("en"),
            checkpoint,
        }
    }

    fn ready() -> Result<()> {
        Ok(())
    }

    /// A padded batch encodes each single-window row as the row alone does,
    /// whatever lengths share it. Rows of one length batch bit for bit on
    /// the dev Mac; padding regroups the CPU softmax's vectorized sum (its
    /// tail joins the SIMD lanes), so padded rows agree within a few F32
    /// ulps of the output's scale.
    #[test]
    fn batched_encode_matches_solo_encode_for_mixed_lengths() {
        let device = Device::Cpu;
        let model = forced_label_model(&device);
        let lengths = [1usize, 3, 37, 64, 125];
        let features = lengths
            .iter()
            .zip(31..)
            .map(|(len, seed)| features_of_frames(&model, *len, seed))
            .collect::<Vec<_>>();
        let solo = features
            .iter()
            .map(|features| model.encode_host_features(features, None, &ready).unwrap())
            .collect::<Vec<_>>();
        let scale = solo
            .iter()
            .map(|artifact| {
                let values = artifact.encoded.flatten_all().unwrap().to_vec1::<f32>();
                values
                    .unwrap()
                    .into_iter()
                    .fold(0f32, |max, value| max.max(value.abs()))
            })
            .fold(0f32, f32::max);
        let mut worst = 0f32;
        for batch in [&[0usize, 1, 2, 3, 4][..], &[2, 3], &[4, 0, 1]] {
            let rows = batch
                .iter()
                .map(|index| batch_row(&features[*index], &ready))
                .collect::<Vec<_>>();
            let batched = model.prepare_retained_encoder_batch(&rows).unwrap();
            assert_eq!(batched.len(), batch.len());
            for (index, artifact) in batch.iter().zip(batched) {
                let artifact = artifact.unwrap();
                assert_eq!(artifact.encoded.dims(), solo[*index].encoded.dims());
                assert_eq!(artifact.encoded_len, lengths[*index]);
                assert_eq!(artifact.encoder_windows, 1);
                assert_eq!(artifact.language.as_deref(), Some("en"));
                assert_eq!(artifact.feature_frames, solo[*index].feature_frames);
                // Its own exact storage, not a view pinning the batch buffer.
                assert_eq!(
                    artifact.resident_tensor_bytes().unwrap(),
                    (lengths[*index] * TINY.dim * 4) as u64
                );
                worst = worst.max(max_abs_diff(&artifact.encoded, &solo[*index].encoded));
            }
        }
        eprintln!(
            "Parakeet batched vs solo encode on CPU: max |diff| {worst:e} over max |x| {scale} ({})",
            if worst == 0.0 {
                "bit-exact"
            } else {
                "not bit-exact"
            }
        );
        // A padded row's softmax sums over the padded length, which the SIMD
        // reduction regroups: 7e-7 of scale here, and another CPU (x86 CI)
        // can regroup differently, so the bound leaves room.
        assert!(
            worst <= BATCHED_ENCODE_TOLERANCE * scale.max(1.0),
            "max |diff| {worst} of {scale}"
        );
    }

    /// The fused attention's wiring: with the kernel's host reference in
    /// place of the eager core (biases, position order, scale and each
    /// row's key length as the CUDA kernel receives them), the conformer
    /// layers match the eager layers on the real frames of a padded batch
    /// and of a lone row. The kernel takes head dims in multiples of 32, so
    /// this encoder is wider than `TINY`.
    #[test]
    fn fused_attention_matches_the_eager_layers() {
        const WIDE: EncoderGeometry = EncoderGeometry {
            mel_bins: 128,
            subsampling_channels: 4,
            layers: 2,
            dim: 64,
            heads: 2,
            ff_dim: 64,
            conv_kernel: 9,
        };
        let device = Device::Cpu;
        let encoder = random_encoder(WIDE, 71, &device);
        let lengths = [29usize, 17, 1];
        let padded = 29;
        let padding = EncoderPadding::new(&lengths, padded, DType::F32, &device).unwrap();
        let mut rng = XorShift(0x5EED_F00D);
        let shape = (lengths.len(), padded, WIDE.dim);
        let x = Tensor::from_vec(
            rng.values(shape.0 * shape.1 * shape.2, 1.0, 0.0),
            shape,
            &device,
        )
        .unwrap()
        .broadcast_mul(&padding.frames.transpose(1, 2).unwrap())
        .unwrap();
        let pos = encoder.positions(padded, &device).unwrap();
        let run = |fused: bool, x: &Tensor, padding: Option<&EncoderPadding>| {
            FUSED_ATTENTION_ON_THE_HOST.with(|flag| flag.set(fused));
            let out = encoder.layers_forward(x.clone(), &pos, padding);
            FUSED_ATTENTION_ON_THE_HOST.with(|flag| flag.set(false));
            out.unwrap()
        };
        let eager = run(false, &x, Some(&padding));
        let fused = run(true, &x, Some(&padding));
        let scale = eager
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let mut worst = 0f32;
        for (row, len) in lengths.iter().enumerate() {
            let real = |t: &Tensor| t.narrow(0, row, 1).unwrap().narrow(1, 0, *len).unwrap();
            worst = worst.max(max_abs_diff(&real(&fused), &real(&eager)));
        }
        let alone = x.narrow(0, 0, 1).unwrap();
        worst = worst.max(max_abs_diff(
            &run(true, &alone, None),
            &run(false, &alone, None),
        ));
        eprintln!("Parakeet fused vs eager attention on CPU: max |diff| {worst:e} of {scale}");
        assert!(
            worst <= 1e-5 * scale.max(1.0),
            "max |diff| {worst} of {scale}"
        );
    }

    /// The padded batch on Metal: the fused depthwise kernel reads the masked
    /// frames, and each row's output is an exact-size copy.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_batched_encode_matches_solo_encode() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_batched_encode_matches_solo_encode")
        else {
            return;
        };
        let model = forced_label_model(&device);
        let lengths = [3usize, 37, 64];
        let features = lengths
            .iter()
            .zip(51..)
            .map(|(len, seed)| features_of_frames(&model, *len, seed))
            .collect::<Vec<_>>();
        let rows = features
            .iter()
            .map(|features| batch_row(features, &ready))
            .collect::<Vec<_>>();
        let batched = model.prepare_retained_encoder_batch(&rows).unwrap();
        for ((features, len), artifact) in features.iter().zip(lengths).zip(batched) {
            let artifact = artifact.unwrap();
            let solo = model.encode_host_features(features, None, &ready).unwrap();
            let scale = host_f32(&solo.encoded)
                .unwrap()
                .into_iter()
                .fold(0f32, |max, value| max.max(value.abs()));
            let diff = max_abs_diff(&artifact.encoded, &solo.encoded);
            assert!(diff <= 1e-5 * scale.max(1.0), "{len}: {diff:e} of {scale}");
            assert_eq!(
                artifact.resident_tensor_bytes().unwrap(),
                (len * TINY.dim * 4) as u64
            );
        }
    }

    /// Padded frames reach no real frame: poisoning them with ±1e3 after the
    /// stem leaves every real frame's output bit for bit, and every padded
    /// frame stays finite. The same layers without the masks let it in.
    #[test]
    fn padded_frames_never_reach_real_frames() {
        let device = Device::Cpu;
        let encoder = random_encoder(TINY, 0xBAD5, &device);
        let lengths = [37usize, 64, 5];
        let padded = 64;
        let stems = lengths
            .iter()
            .zip(7..)
            .map(|(len, seed)| {
                let values = XorShift(seed).values(len * TINY.dim, 1.0, 0.0);
                Tensor::from_vec(values, (1, *len, TINY.dim), &device).unwrap()
            })
            .collect::<Vec<_>>();
        let batch = |poison: f32| {
            let rows = stems
                .iter()
                .zip(lengths)
                .map(|(stem, len)| {
                    if len == padded {
                        return stem.clone();
                    }
                    let fill = (0..(padded - len) * TINY.dim)
                        .map(|index| if index % 2 == 0 { poison } else { -poison })
                        .collect::<Vec<_>>();
                    let fill = Tensor::from_vec(fill, (1, padded - len, TINY.dim), &device);
                    Tensor::cat(&[stem, &fill.unwrap()], 1).unwrap()
                })
                .collect::<Vec<_>>();
            Tensor::cat(&rows, 0).unwrap()
        };
        let clean = encoder.encode_padded(batch(0.0), &lengths).unwrap();
        let poisoned = encoder.encode_padded(batch(1e3), &lengths).unwrap();
        let real = |x: &Tensor, row: usize| {
            x.narrow(0, row, 1)
                .unwrap()
                .narrow(1, 0, lengths[row])
                .unwrap()
        };
        for row in 0..lengths.len() {
            assert_eq!(
                real(&clean, row).to_vec3::<f32>().unwrap(),
                real(&poisoned, row).to_vec3::<f32>().unwrap(),
                "row {row}"
            );
        }
        for x in [&clean, &poisoned] {
            let values = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert!(values.iter().all(|value| value.is_finite()));
        }

        let pos = build_rel_positional_embedding(padded, TINY.dim, &device).unwrap();
        let mut unmasked = batch(1e3);
        for layer in &encoder.layers {
            unmasked = layer.forward(&unmasked, &pos, None).unwrap();
        }
        assert!(max_abs_diff(&real(&unmasked, 0), &real(&clean, 0)) > 1e-3);
    }

    /// A batch of one request runs today's windowed encode, bit for bit,
    /// whether its input fits one window or not; a multi-window input never
    /// joins a wider batch.
    #[test]
    fn width_one_batch_is_the_solo_encode() {
        let device = Device::Cpu;
        let mut model = forced_label_model(&device);
        // 4 s: 50 encoder frames.
        let audio = XorShift(13).values(64_037, 0.1, 0.0);
        let windowed = EncoderWindowing {
            frames: 32,
            left_context: 10,
            right_context: 10,
        };
        for (windowing, windows) in [(EncoderWindowing::V3, 1), (windowed, 3)] {
            model.encoder_windowing = windowing;
            let features = model.prepare_host_features(&audio, 16_000).unwrap();
            assert_eq!(features.single_window(), windows == 1);
            let batched = model
                .prepare_retained_encoder_batch(&[batch_row(&features, &ready)])
                .unwrap()
                .pop()
                .unwrap()
                .unwrap();
            let solo = model
                .prepare_retained_encoder(&audio, 16_000, Some("en"), &ready)
                .unwrap();
            assert_eq!(batched.encoder_windows, windows);
            assert_eq!(
                batched.encoded.to_vec3::<f32>().unwrap(),
                solo.encoded.to_vec3::<f32>().unwrap()
            );
        }
        let multi_window = model.prepare_host_features(&audio, 16_000).unwrap();
        let single_window = features_of_frames(&model, 20, 3);
        let mixed = [
            batch_row(&single_window, &ready),
            batch_row(&multi_window, &ready),
        ];
        assert!(model.prepare_retained_encoder_batch(&mixed).is_err());
    }

    /// A row cancelled before the encode fails alone: the others encode as
    /// they would have without it, and a lone survivor runs the solo encode.
    #[test]
    fn a_cancelled_batch_row_fails_alone() {
        let model = forced_label_model(&Device::Cpu);
        let features = [5usize, 7, 6]
            .into_iter()
            .zip(41..)
            .map(|(len, seed)| features_of_frames(&model, len, seed))
            .collect::<Vec<_>>();
        let cancelled = || -> Result<()> { Err(Error::Cancelled("parakeet-row".into())) };
        let solo = |index: usize| {
            model
                .encode_host_features(&features[index], None, &ready)
                .unwrap()
        };

        let rows = [
            batch_row(&features[0], &ready),
            batch_row(&features[1], &cancelled),
            batch_row(&features[2], &ready),
        ];
        let outcomes = model.prepare_retained_encoder_batch(&rows).unwrap();
        assert!(matches!(outcomes[1], Err(Error::Cancelled(_))));
        for index in [0, 2] {
            let artifact = outcomes[index].as_ref().unwrap();
            let alone = solo(index).encoded;
            let scale = alone
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(
                max_abs_diff(&artifact.encoded, &alone)
                    <= BATCHED_ENCODE_TOLERANCE * scale.max(1.0)
            );
        }

        let rows = [
            batch_row(&features[0], &cancelled),
            batch_row(&features[1], &ready),
            batch_row(&features[2], &cancelled),
        ];
        let outcomes = model.prepare_retained_encoder_batch(&rows).unwrap();
        assert!(matches!(outcomes[0], Err(Error::Cancelled(_))));
        assert!(matches!(outcomes[2], Err(Error::Cancelled(_))));
        assert_eq!(
            outcomes[1]
                .as_ref()
                .unwrap()
                .encoded
                .to_vec3::<f32>()
                .unwrap(),
            solo(1).encoded.to_vec3::<f32>().unwrap()
        );

        let outcomes = model
            .prepare_retained_encoder_batch(&[batch_row(&features[0], &cancelled)])
            .unwrap();
        assert!(matches!(outcomes[..], [Err(Error::Cancelled(_))]));
    }

    /// The pre-change per-head argmax: one scalar readback per head.
    fn reference_argmax(x: &Tensor) -> usize {
        x.argmax(candle_core::D::Minus1)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap() as usize
    }

    #[test]
    fn parakeet_tdt_argmax_selects_half_logits_on_device() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, -0.3, 0.8, 0.2, 0.5, 0.9, -1.0], (7,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        assert_eq!(tdt_label_and_duration(&logits, 4).unwrap(), (2, 1));
    }

    #[test]
    fn parakeet_tdt_argmax_rejects_missing_heads() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.2], (2,), &device).unwrap();
        assert!(tdt_label_and_duration(&logits, 2).is_err());
        assert!(tdt_label_and_duration(&logits, 0).is_err());
        let empty = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        assert!(tdt_label_and_duration(&empty, 0).is_err());
    }

    /// One combined readback selects the same label and duration as the two
    /// per-head argmax readbacks it replaces, for single rows and batches.
    #[test]
    fn combined_tdt_readback_matches_per_head_argmax() {
        let device = Device::Cpu;
        let (token_end, durations, rows) = (8193, 5, 3);
        let values = XorShift(0x7DD4_1A2B).values(rows * (token_end + durations), 8.0, 0.0);
        let logits = Tensor::from_vec(values, (rows, token_end + durations), &device).unwrap();
        let batch = tdt_batch_labels_and_durations(&logits, token_end, durations).unwrap();
        for (row, &selected) in batch.iter().enumerate() {
            let row_logits = logits.get(row).unwrap();
            let expected = (
                reference_argmax(&row_logits.narrow(0, 0, token_end).unwrap()),
                reference_argmax(&row_logits.narrow(0, token_end, durations).unwrap()),
            );
            assert_eq!(selected, expected);
            assert_eq!(
                tdt_label_and_duration(&row_logits, token_end).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn predictor_native_batch_matches_independent_scalar_rows() {
        let device = Device::Cpu;
        let predictor = zero_predictor(&device);
        let labels = [0usize, predictor.blank_idx];
        let mut batch_state = predictor.initial_state(labels.len(), &device).unwrap();
        let batch = predictor
            .step_batch(&labels, &mut batch_state, &device)
            .unwrap();

        let mut scalar_outputs = Vec::new();
        let mut scalar_states = Vec::new();
        for label in labels {
            let mut state = predictor.initial_state(1, &device).unwrap();
            scalar_outputs.push(predictor.step(label, &mut state, &device).unwrap());
            scalar_states.push(state);
        }
        let output_refs = scalar_outputs.iter().collect::<Vec<_>>();
        let expected = Tensor::cat(&output_refs, 0).unwrap();
        assert_eq!(
            batch.to_vec3::<f32>().unwrap(),
            expected.to_vec3::<f32>().unwrap()
        );
        for (batched, scalar_component) in [
            (
                &batch_state.h0,
                scalar_states
                    .iter()
                    .map(|state| &state.h0)
                    .collect::<Vec<_>>(),
            ),
            (
                &batch_state.c0,
                scalar_states
                    .iter()
                    .map(|state| &state.c0)
                    .collect::<Vec<_>>(),
            ),
            (
                &batch_state.h1,
                scalar_states
                    .iter()
                    .map(|state| &state.h1)
                    .collect::<Vec<_>>(),
            ),
            (
                &batch_state.c1,
                scalar_states
                    .iter()
                    .map(|state| &state.c1)
                    .collect::<Vec<_>>(),
            ),
        ] {
            let expected = Tensor::cat(&scalar_component, 0).unwrap();
            assert_eq!(
                batched.to_vec2::<f32>().unwrap(),
                expected.to_vec2::<f32>().unwrap()
            );
        }

        let scalar_refs = scalar_states.iter().collect::<Vec<_>>();
        let gathered = super::ParakeetPredictorBatchState::gather_rows(&scalar_refs).unwrap();
        assert_eq!(gathered.h0.dims(), batch_state.h0.dims());
        let mut scattered = [
            predictor.initial_state(1, &device).unwrap(),
            predictor.initial_state(1, &device).unwrap(),
        ];
        let mut targets = scattered.iter_mut().collect::<Vec<_>>();
        gathered.scatter_rows(&mut targets).unwrap();
        assert_eq!(
            scattered[0].h0.to_vec2::<f32>().unwrap(),
            scalar_states[0].h0.to_vec2::<f32>().unwrap()
        );
        let before = scattered[0].h0.to_vec2::<f32>().unwrap();
        let mut short_targets = vec![&mut scattered[0]];
        assert!(gathered.scatter_rows(&mut short_targets).is_err());
        assert_eq!(scattered[0].h0.to_vec2::<f32>().unwrap(), before);
    }

    #[test]
    fn joint_native_batch_matches_independent_scalar_rows() {
        let device = Device::Cpu;
        let enc_hidden = 3usize;
        let output_dim = 5usize;
        let joint = Joint {
            pred: Linear::new(
                Tensor::zeros((PRED_HIDDEN, PRED_HIDDEN), DType::F32, &device).unwrap(),
                None,
            ),
            enc: Linear::new(
                Tensor::zeros((PRED_HIDDEN, enc_hidden), DType::F32, &device).unwrap(),
                None,
            ),
            out: Linear::new(
                Tensor::zeros((output_dim, PRED_HIDDEN), DType::F32, &device).unwrap(),
                Some(Tensor::arange(0f32, output_dim as f32, &device).unwrap()),
            ),
            num_classes_with_blank: 3,
            num_durations: 2,
        };
        let encoded = Tensor::zeros((2, enc_hidden), DType::F32, &device).unwrap();
        let predictor = Tensor::zeros((2, 1, PRED_HIDDEN), DType::F32, &device).unwrap();
        let batch = joint.joint_batch_rows(&encoded, &predictor).unwrap();
        let mut scalar = Vec::new();
        for row in 0..2 {
            scalar.push(
                joint
                    .joint_batch_rows(
                        &encoded.narrow(0, row, 1).unwrap(),
                        &predictor.narrow(0, row, 1).unwrap(),
                    )
                    .unwrap(),
            );
        }
        let refs = scalar.iter().collect::<Vec<_>>();
        let expected = Tensor::cat(&refs, 0).unwrap();
        assert_eq!(
            batch.to_vec2::<f32>().unwrap(),
            expected.to_vec2::<f32>().unwrap()
        );
    }

    /// A graph's shape: rows up to the next graph width, frames up to a
    /// power of two, and nothing past 32 rows, one window or the pool.
    #[test]
    fn encoder_graph_shapes_pad_rows_to_a_width_and_frames_to_a_power_of_two() {
        assert_eq!(encoder_graph_shape(1, 125), Some((1, 128)));
        assert_eq!(encoder_graph_shape(0, 0), Some((1, 1)));
        assert_eq!(encoder_graph_shape(5, 64), Some((6, 64)));
        assert_eq!(encoder_graph_shape(17, 100), Some((24, 128)));
        // Every width of 10 s clips, 12 rows of 20 s, a lone 82 s window.
        assert_eq!(encoder_graph_shape(32, 125), Some((32, 128)));
        assert_eq!(encoder_graph_shape(12, 250), Some((12, 256)));
        assert_eq!(encoder_graph_shape(1, 1000), Some((1, 1024)));
        assert_eq!(encoder_graph_shape(33, 8), None);
        assert_eq!(encoder_graph_shape(13, 250), None, "16 rows of 256 frames");
        assert_eq!(encoder_graph_shape(2, 1000), None);
        assert_eq!(encoder_graph_shape(1, 1025), None, "past one window");
        for rows in ENCODER_GRAPH_ROWS {
            for frames in (0..=10).map(|exp| 1usize << exp) {
                let Some(shape) = encoder_graph_shape(rows, frames) else {
                    continue;
                };
                assert_eq!(shape, (rows, frames), "a graph's shape is its own");
                let bytes = encoder_window_batch_bytes(frames, rows, BackendKind::Cuda).unwrap();
                assert!(bytes.layers <= ENCODER_GRAPH_LIMITS.pool_bytes);
            }
        }
    }

    /// With the fused attention on, a window prices its activations and one
    /// positional projection, not the scores, the three `rel_shift` planes
    /// or a projection copy per row: 224 MiB less for a lone 1,024-frame
    /// window, and about a third of the eager working set left.
    #[test]
    fn fused_attention_drops_the_eager_attention_planes() {
        for rows in [1usize, 4] {
            let eager =
                encoder_window_batch_bytes_with(1024, rows, BackendKind::Cuda, false).unwrap();
            let fused =
                encoder_window_batch_bytes_with(1024, rows, BackendKind::Cuda, true).unwrap();
            let rows64 = rows as u64;
            let planes = rows64 * 8 * 1024 * 1024 * 4 + 3 * rows64 * 8 * 1024 * 2048 * 4;
            let copies = if rows > 1 {
                rows64 * 1024 * 2048 * 4
            } else {
                0
            };
            assert_eq!(fused.stem, eager.stem);
            assert_eq!(
                eager.layers - fused.layers,
                planes + copies - 1024 * 2048 * 4
            );
            assert!(fused.layers * 3 < eager.layers, "{fused:?} vs {eager:?}");
        }
        // Without the CUDA kernel's verdict, pricing stays eager.
        assert_eq!(
            encoder_window_batch_bytes(128, 2, BackendKind::Cuda).unwrap(),
            encoder_window_batch_bytes_with(128, 2, BackendKind::Cuda, false).unwrap()
        );
    }

    /// Admission prices the graphed shape on CUDA only, each real row
    /// carrying its share of the padded batch; the graphs' own memory is a
    /// CUDA load's.
    #[test]
    fn cuda_admission_prices_the_graphed_shape() {
        // Pricing and the encoder read `IZWI_PARAKEET_ENCODER_GRAPHS`.
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(parakeet_encoder_graph_device_bytes(BackendKind::Cpu), 0);
        assert_eq!(parakeet_encoder_graph_device_bytes(BackendKind::Metal), 0);
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(encoder_batch_shape(5, 100, backend), (5, 100));
        }
        if !encoder_graphs_requested() {
            assert_eq!(parakeet_encoder_graph_device_bytes(BackendKind::Cuda), 0);
            assert_eq!(encoder_batch_shape(5, 100, BackendKind::Cuda), (5, 100));
            return;
        }
        assert_eq!(
            parakeet_encoder_graph_device_bytes(BackendKind::Cuda),
            768 << 20
        );
        assert_eq!(encoder_batch_shape(5, 100, BackendKind::Cuda), (6, 128));
        assert_eq!(encoder_batch_shape(13, 250, BackendKind::Cuda), (13, 250));

        let model = forced_label_model(&Device::Cpu);
        let features = features_of_frames(&model, 38, 5);
        for rows in [1usize, 3, 5, 17, 32, 40] {
            let (batch_rows, frames) = encoder_batch_shape(rows, 38, BackendKind::Cuda);
            let batch = encoder_window_batch_bytes(frames, batch_rows, BackendKind::Cuda).unwrap();
            let row = parakeet_encode_row_bytes(&features, 38, rows, BackendKind::Cuda).unwrap();
            let eager = parakeet_encode_row_bytes(&features, 38, rows, BackendKind::Cpu).unwrap();
            assert!(row.window_bytes * rows as u64 >= batch.stem + batch.layers);
            assert!(row.window_bytes >= eager.window_bytes, "{rows} rows");
            let (useful, materialized) =
                parakeet_encode_row_elements(&features, 38, rows, BackendKind::Cuda).unwrap();
            assert_eq!(useful, (38 * EncoderGeometry::V3.dim) as u64);
            assert_eq!(materialized, (frames * EncoderGeometry::V3.dim) as u64);
        }
    }

    /// `IZWI_PARAKEET_ENCODER_GRAPHS=0` reaches admission (no graph memory,
    /// the eager shape) and the loader (no graphs, naming the switch).
    #[test]
    fn the_kill_switch_turns_the_graphs_and_their_pricing_off() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os(PARAKEET_ENCODER_GRAPHS_ENV);
        std::env::set_var(PARAKEET_ENCODER_GRAPHS_ENV, "0");
        let bytes = parakeet_encoder_graph_device_bytes(BackendKind::Cuda);
        let shape = encoder_batch_shape(5, 100, BackendKind::Cuda);
        let reason = random_encoder(TINY, 1, &Device::Cpu).graphs.err();
        std::env::remove_var(PARAKEET_ENCODER_GRAPHS_ENV);
        let on = (
            parakeet_encoder_graph_device_bytes(BackendKind::Cuda),
            encoder_batch_shape(5, 100, BackendKind::Cuda),
        );
        if let Some(previous) = previous {
            std::env::set_var(PARAKEET_ENCODER_GRAPHS_ENV, previous);
        }
        assert_eq!(bytes, 0);
        assert_eq!(shape, (5, 100));
        let reason = reason.expect("no graphs");
        assert!(reason.contains(PARAKEET_ENCODER_GRAPHS_ENV), "{reason}");
        assert_eq!(on, (768 << 20, (6, 128)), "on when unset");
    }

    /// Off CUDA the encoder has no graphs, and says why.
    #[test]
    fn cpu_encoder_runs_without_graphs() {
        let encoder = random_encoder(TINY, 1, &Device::Cpu);
        assert!(!encoder.graphs_on());
        assert_eq!(encoder.batch_shape(5, 100), (5, 100));
        let off = encoder.graphs_diagnostics()["off"].clone();
        assert!(
            off.as_str().is_some_and(|reason| !reason.is_empty()),
            "{off}"
        );
    }

    /// On CUDA, a padded batch (spare rows included) and a lone window run
    /// through their graphs: capture, verify, then two replays, each bit for
    /// bit the eager layers at the same padded shape, over fresh inputs.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_encoder_graphs_replay_the_eager_layers_bit_for_bit() {
        // Pricing and the encoder read `IZWI_PARAKEET_ENCODER_GRAPHS`.
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        if !encoder_graphs_requested() {
            return;
        }
        let mut model = forced_label_model(&device);
        model.network.encoder = bf16_encoder(1, &device);
        assert!(
            model.network.encoder.graphs_on(),
            "{}",
            model.network.encoder.graphs_diagnostics()
        );
        let mut eager = bf16_encoder(1, &device);
        eager.graphs = Err("eager reference".into());
        let bits = |t: &Tensor| -> Vec<u32> {
            host_f32(t).unwrap().into_iter().map(f32::to_bits).collect()
        };
        let upload = |features: &ParakeetHostFeatures| {
            let len = features.encoded_len();
            let columns = EncoderWindow::whole(len).feature_columns(features.feature_frames());
            let window = features
                .log_mel
                .normalize_and_upload(columns, &device)
                .unwrap();
            (window, len)
        };
        for step in 0..4u64 {
            // Five rows pad to the six-row graph of 64 frames.
            let lengths = [37usize, 64, 50, 12, 61];
            let features = lengths
                .iter()
                .zip(0..)
                .map(|(len, index)| features_of_frames(&model, *len, 100 * step + index + 1))
                .collect::<Vec<_>>();
            let rows = features
                .iter()
                .map(|features| batch_row(features, &ready))
                .collect::<Vec<_>>();
            let graphed = model.prepare_retained_encoder_batch(&rows).unwrap();
            let uploaded = features.iter().map(upload).collect::<Vec<_>>();
            assert_eq!(model.network.encoder.batch_shape(5, 64), (6, 64));
            let want = eager
                .encode_single_window_batch_at(&uploaded, (6, 64))
                .unwrap();
            for (index, (got, want)) in graphed.into_iter().zip(&want).enumerate() {
                let got = got.unwrap();
                assert_eq!(got.encoded.dims(), want.dims());
                assert_eq!(bits(&got.encoded), bits(want), "step {step} row {index}");
            }

            // A lone 125-frame window runs the one-row graph of 128 frames.
            let solo = features_of_frames(&model, 125, 7 + step);
            let got = model.encode_host_features(&solo, None, &ready).unwrap();
            let want = eager
                .encode_single_window_batch_at(&[upload(&solo)], (1, 128))
                .unwrap();
            assert_eq!(bits(&got.encoded), bits(&want[0]), "step {step} solo");
        }
        let diagnostics = model.network.encoder.graphs_diagnostics();
        assert!(diagnostics["off"].is_null(), "{diagnostics}");
        assert_eq!(diagnostics["captures"], 2, "{diagnostics}");
        assert_eq!(diagnostics["verified"], 2, "{diagnostics}");
        assert_eq!(diagnostics["replays"], 4, "{diagnostics}");
        assert_eq!(diagnostics["fallbacks"], 0, "{diagnostics}");
    }
}
