mod decode;
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
use crate::kernels::depthwise_conv::{depthwise_conv1d, DepthwiseConv1dSpec};
use crate::kernels::f16_weight_linear::{
    load_weight_only_f16, PointwiseConv1d, ProjectionLinear, WeightOnlyF16Load,
    F16_PROJECTION_MAX_RELATIVE_L2,
};
use crate::kernels::numeric_check::{
    run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};
use crate::kv::v2::{
    ComponentShapeInstantiation, DomainStepIntent, ShapeAxis, ShapeDimensionValue,
    StateComponentId, StateUpdateKind,
};
use crate::model::ModelVariant;
use crate::models::shared::detokenize::IncrementalDetokenizer;
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::weights::mlx;
use crate::tokenizer::Tokenizer;

use decode::decode_tokens;
use nemo::{ensure_parakeet_artifacts, ParakeetArtifacts};
pub(crate) use nemo::{parakeet_checkpoint_paths, parakeet_native_artifacts_complete};
#[cfg(test)]
pub(crate) use physical::{parakeet_physical_state_spec, PARAKEET_PREDICTOR_STATE_DOMAIN};
pub(crate) use physical::{ParakeetPhysicalStateSpec, PARAKEET_PREDICTOR_STATE_GROUP};
use preprocessor::{ParakeetLogMel, ParakeetPreprocessor};

/// The dtype the Parakeet loader reads every weight in, and the activation
/// dtype on every backend. On Metal the encoder's projection weights are then
/// stored in F16 (see [`resolve_parakeet_projection_precision`]). Load
/// admission prices the checkpoint from both
/// (`runtime/lifecycle/asr_memory.rs`).
pub(crate) const PARAKEET_WEIGHT_DTYPE: DType = DType::F32;

/// Parakeet's dtype override. On Metal, `f32` keeps every weight in F32 and
/// `f16` stores the encoder projections in F16; see
/// [`resolve_parakeet_projection_precision`].
pub(crate) const PARAKEET_ASR_DTYPE_ENV: &str = "IZWI_PARAKEET_ASR_DTYPE";

/// Whether the encoder's projection weights (FFN, attention q/k/v/out/pos,
/// pointwise convs and the subsampling output projection) load in F16 on
/// `device`: F16 when they do, F32 when every weight stays F32. The rest
/// stays F32 as in whisper.cpp's Parakeet converter: subsampling convs,
/// depthwise conv and BatchNorm, LayerNorms, biases, positional biases, and
/// the TDT predictor and joint. Activations are F32 throughout. A failed
/// load-time self-check still falls back to F32 ([`load_weight_only_f16`]).
pub(crate) fn resolve_parakeet_projection_precision(device: &DeviceProfile) -> DTypeSelection {
    let family = std::env::var(PARAKEET_ASR_DTYPE_ENV).ok();
    device.resolve_metal_weight_only_f16(
        family
            .as_deref()
            .map(|value| (PARAKEET_ASR_DTYPE_ENV, value)),
    )
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

/// The host-side builder for weights that load in F16: they are read on the
/// CPU and converted there, so their F32 form never reaches the GPU.
fn projection_vb<'a>(vb: &VarBuilder<'a>, f16: Option<&Device>) -> VarBuilder<'a> {
    match f16 {
        Some(_) => vb.clone().set_device(Device::Cpu),
        None => vb.clone(),
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
            .then_some(EncoderWindow {
                start: 0,
                end: encoded_len,
                keep_start: 0,
                keep_end: encoded_len,
            })
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

/// Peak preparation memory beyond the weights and the retained artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParakeetPreparationBytes {
    /// Host log-mel pipeline: mono copy, pre-emphasis and padded copies,
    /// power spectrum and log-mel.
    pub(crate) host_bytes: u64,
    /// One encoder window's peak: the first subsampling convolution and its
    /// ReLU copy, the attention scores and `rel_shift` buffers, and the
    /// per-frame activations of a layer.
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
    const F32_BYTES: u64 = 4;
    const HOST_BYTES_PER_SAMPLE: u64 = 22;
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
    let valid_frames = preprocessor::valid_feature_frames(samples);
    let encoded_len = as_u64(subsampled_len_3x(valid_frames))?;
    // The STFT emits one padding frame past the valid ones.
    let window = as_u64(subsampled_len_3x(valid_frames + 1).min(EncoderWindowing::V3.frames))?;
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
    // conv0 output [channels, 8T/2, mels/2] and its ReLU copy.
    let subsampling = multiply(&[2, channels, 4 * window, mel_bins / 2, F32_BYTES])?;
    let scores = multiply(&[heads, window, window, F32_BYTES])?;
    // q_v·pᵀ, its zero pad and the reshaped copy, each [H, T, 2T].
    let rel_shift = multiply(&[heads, window, 2 * window, F32_BYTES])?;
    let activation = multiply(&[window, ff_dim, F32_BYTES])?;
    let window_bytes = [
        round(subsampling)?,
        round(scores)?,
        multiply(&[3, round(rel_shift)?])?,
        multiply(&[LAYER_ACTIVATIONS, round(activation)?])?,
    ]
    .into_iter()
    .try_fold(0u64, |total, bytes| total.checked_add(bytes))
    .ok_or_else(overflow)?;
    Ok(ParakeetPreparationBytes {
        host_bytes: multiply(&[as_u64(samples)?, HOST_BYTES_PER_SAMPLE])?,
        window_bytes,
        output_bytes: round(multiply(&[encoded_len, dim, F32_BYTES])?)?,
    })
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
    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        let mut accounting = TensorStorageAccounting::default();
        accounting.add_tensor(&self.encoded).ok_or_else(|| {
            Error::Overloaded("Parakeet encoder artifact storage accounting overflowed".into())
        })?;
        Ok(accounting.bytes())
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
    t: usize,
    last_emit_t: usize,
    emit_count_at_t: usize,
    guard_steps: usize,
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
    pub(crate) fn prepare_retained_encoder(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        checkpoint: &dyn Fn() -> Result<()>,
    ) -> Result<ParakeetPreparedEncoderArtifact> {
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
        let encode_started = Instant::now();
        let encoded = self.encode_log_mel(&log_mel, &mut ParakeetTimings::default(), checkpoint)?;
        let encode_ms = elapsed_ms(encode_started);
        // Preparation sized the decode budget from the sample count alone.
        let budgeted_len = parakeet_encoded_len_for_samples(audio.len(), sample_rate);
        if encoded.encoded_len != budgeted_len {
            return Err(Error::InferenceError(format!(
                "Parakeet encoder produced {} frames but its decode budget assumed {budgeted_len}",
                encoded.encoded_len
            )));
        }
        Ok(ParakeetPreparedEncoderArtifact {
            encoded: deep_copy_tensor_storage(&encoded.encoded)?,
            encoded_len: encoded.encoded_len,
            input_samples: audio.len(),
            input_sample_rate: sample_rate,
            resampled_samples,
            feature_frames: log_mel.valid_frames(),
            language: language.map(ToOwned::to_owned),
            windowing: self.encoder_windowing,
            encoder_windows: encoded.windows,
            log_mel_ms,
            encode_ms,
        })
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
                    t: 0,
                    last_emit_t: usize::MAX,
                    emit_count_at_t: 0,
                    guard_steps: 0,
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
            "weight_precision": self.weight_precision.diagnostics(),
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
        let result = (|| {
            loop {
                let encoded_rows = rows
                    .iter()
                    .map(|row| {
                        if row.state.t >= row.artifact.encoded_len {
                            return Err(Error::InferenceError(
                                "Parakeet retained cursor exceeded encoder artifact".into(),
                            ));
                        }
                        row.artifact
                            .encoded
                            .i((0, row.state.t, ..))?
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
                for (index, row) in rows.iter_mut().enumerate() {
                    let (label, duration) = selections[index];
                    let duration = duration.min(self.num_durations.saturating_sub(1));
                    let mut jump = duration;
                    if label == self.blank_idx && jump == 0 {
                        jump = 1;
                    }
                    let t_cur = row.state.t;
                    row.state.t = row.state.t.saturating_add(jump);
                    row.state.guard_steps = row.state.guard_steps.saturating_add(1);
                    row.state.counters.tdt_joint_steps =
                        row.state.counters.tdt_joint_steps.saturating_add(1);
                    row.state.counters.device_argmax_scalar_reads = row
                        .state
                        .counters
                        .device_argmax_scalar_reads
                        .saturating_add(1);
                    if label == self.blank_idx {
                        row.state.counters.tdt_blank_steps =
                            row.state.counters.tdt_blank_steps.saturating_add(1);
                        continue;
                    }
                    if t_cur == row.state.last_emit_t {
                        row.state.emit_count_at_t = row.state.emit_count_at_t.saturating_add(1);
                    } else {
                        row.state.last_emit_t = t_cur;
                        row.state.emit_count_at_t = 1;
                    }
                    if row.state.emit_count_at_t > self.max_symbols {
                        row.state.t = t_cur.saturating_add(1);
                        continue;
                    }
                    row.state.token_ids.push(label);
                    row.state.token_frames.push(TdtTokenFrame {
                        frame: t_cur,
                        duration,
                    });
                    emitted.push((index, label));
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
                    let exhausted = row.state.t >= row.artifact.encoded_len
                        || tokens >= row.artifact.encoded_len.saturating_mul(self.max_symbols);
                    let budget_reached = !exhausted && tokens >= row.max_tokens.max(1);
                    if budget_reached {
                        row.state.stop_reason = Some("max_tokens");
                    }
                    row.state.finished =
                        exhausted || budget_reached || row.state.guard_steps >= guard_limit;
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
            if weight_precision.projections_f16 {
                "F16"
            } else {
                "F32"
            },
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

        Ok(Self {
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
        })
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
        diagnostics["execution"]["weight_precision"] = self.weight_precision.diagnostics();

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
}

impl ParakeetEncoder {
    /// Loads the encoder with its projections in F16 on `f16` when given.
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(vb.pp("pre_encode"), geometry, f16)?;
        let layers = (0..geometry.layers)
            .map(|idx| ConformerLayer::load(vb.pp(format!("layers.{idx}")), geometry, f16))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            geometry,
            pre_encode,
            layers,
        })
    }

    /// Loads the encoder at `precision` (F16 = weight-only F16 projections),
    /// keeping F16 only if [`Self::weight_only_f16_self_check`] passes.
    fn load_with_precision(
        vb: VarBuilder,
        geometry: EncoderGeometry,
        precision: DTypeSelection,
    ) -> Result<(Self, WeightOnlyF16Load)> {
        load_weight_only_f16(
            "Parakeet",
            &precision,
            &vb.device().clone(),
            |f16| Self::load(vb.clone(), geometry, f16),
            |encoder| encoder.weight_only_f16_self_check(&vb),
        )
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
                    projection_vb(vb, Some(&Device::Cpu)).pp("layers.0.feed_forward1.linear1"),
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
            let pos = build_rel_positional_embedding(x.dim(1)?, geometry.dim, device)?;
            for (index, layer) in self.layers.iter().enumerate() {
                x = layer.forward(&x, &pos)?;
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
        let mut stitched: Option<Tensor> = None;
        let mut pos_emb: Option<Tensor> = None;
        for window in &plan {
            checkpoint()?;
            let upload_started = Instant::now();
            let features = window_features(window.feature_columns(feature_frames))?;
            timings.feature_upload_ms += elapsed_ms(upload_started);

            let frames = window.end - window.start;
            let pre_encode_started = Instant::now();
            let x = self.pre_encode.forward(&features)?;
            drop(features);
            if x.dim(1)? < frames {
                return Err(Error::InferenceError(format!(
                    "Parakeet window produced {} encoder frames, expected {frames}",
                    x.dim(1)?
                )));
            }
            // Attention must see only this window's frames, never one made
            // purely of STFT padding.
            let mut x = x.narrow(1, 0, frames)?;
            timings.pre_encode_ms += elapsed_ms(pre_encode_started);

            // Relative positions depend only on i - j, so one embedding of
            // the window length serves every window.
            let pos_started = Instant::now();
            let pos = match pos_emb.take() {
                Some(pos) if pos.dim(1)? == 2 * frames - 1 => pos,
                _ => build_rel_positional_embedding(frames, self.geometry.dim, x.device())?,
            };
            timings.pos_emb_ms += elapsed_ms(pos_started);

            let layers_started = Instant::now();
            for layer in &self.layers {
                x = layer.forward(&x, &pos)?;
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
}

/// The stitched `[1, E, D]` encoder output.
struct ParakeetEncoded {
    encoded: Tensor,
    encoded_len: usize,
    windows: usize,
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: ProjectionLinear,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
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

        let out = ProjectionLinear::new(
            mlx::load_linear(
                channels * geometry.subsampled_mel_bins(),
                geometry.dim,
                projection_vb(&vb, f16).pp("out"),
            )?,
            f16,
        )?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features: &Tensor) -> Result<Tensor> {
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

        let (b, c, t, f) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)
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
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
        let dim = geometry.dim;
        let norm_ff1 = layer_norm(dim, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"), geometry, f16)?;

        let norm_self_att = layer_norm(dim, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"), geometry, f16)?;

        let norm_conv = layer_norm(dim, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"), geometry, f16)?;

        let norm_ff2 = layer_norm(dim, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"), geometry, f16)?;

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
    linear1: ProjectionLinear,
    linear2: ProjectionLinear,
}

impl FeedForward {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
        let vb = projection_vb(&vb, f16);
        let linear1 = ProjectionLinear::new(
            mlx::load_linear_no_bias(geometry.dim, geometry.ff_dim, vb.pp("linear1"))?,
            f16,
        )?;
        let linear2 = ProjectionLinear::new(
            mlx::load_linear_no_bias(geometry.ff_dim, geometry.dim, vb.pp("linear2"))?,
            f16,
        )?;
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
    dim: usize,
    pointwise_conv1: PointwiseConv1d,
    depthwise_conv: Conv1d,
    folded_depthwise_weight: Tensor,
    folded_depthwise_bias: Tensor,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: PointwiseConv1d,
}

impl ConformerConv {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
        let dim = geometry.dim;
        let pointwise_vb = projection_vb(&vb, f16);
        let pointwise_conv1 = PointwiseConv1d::new(
            mlx::load_conv1d_no_bias(
                dim,
                dim * 2,
                1,
                Conv1dConfig::default(),
                pointwise_vb.pp("pointwise_conv1"),
            )?,
            f16,
        )?;

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

        let pointwise_conv2 = PointwiseConv1d::new(
            mlx::load_conv1d_no_bias(
                dim,
                dim,
                1,
                Conv1dConfig::default(),
                pointwise_vb.pp("pointwise_conv2"),
            )?,
            f16,
        )?;

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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // [B, T, C] -> [B, C, T]
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..self.dim, ..))?;
        let x_b = x.i((.., self.dim.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

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
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
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

struct RelPosSelfAttention {
    heads: usize,
    head_dim: usize,
    linear_q: ProjectionLinear,
    linear_k: ProjectionLinear,
    linear_v: ProjectionLinear,
    linear_out: ProjectionLinear,
    linear_pos: ProjectionLinear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder, geometry: EncoderGeometry, f16: Option<&Device>) -> Result<Self> {
        let dim = geometry.dim;
        let (heads, head_dim) = (geometry.heads, geometry.head_dim());
        let projection_vb = projection_vb(&vb, f16);
        let projection = |name: &str| {
            ProjectionLinear::new(
                mlx::load_linear_no_bias(dim, dim, projection_vb.pp(name))?,
                f16,
            )
            .map_err(Error::from)
        };
        let linear_q = projection("linear_q")?;
        let linear_k = projection("linear_k")?;
        let linear_v = projection("linear_v")?;
        let linear_out = projection("linear_out")?;
        let linear_pos = projection("linear_pos")?;

        // In Parakeet checkpoints these are direct tensors under self_attn.*.
        let pos_bias_u = vb.get((heads, head_dim), "pos_bias_u")?;
        let pos_bias_v = vb.get((heads, head_dim), "pos_bias_v")?;

        Ok(Self {
            heads,
            head_dim,
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let (heads, head_dim) = (self.heads, self.head_dim);

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // [B, H, T, Dh]
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, heads, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, heads, head_dim))?
            .transpose(1, 2)?; // [1, H, 2T-1, Dh]

        let pos_bias_u = self.pos_bias_u.reshape((1, heads, 1, head_dim))?;
        let pos_bias_v = self.pos_bias_v.reshape((1, heads, 1, head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (head_dim as f64).sqrt(), 0.0)?;
        let scores = if scores.is_contiguous() {
            scores
        } else {
            scores.contiguous()?
        };
        let attn = ops::softmax_last_dim(&scores)?;

        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;

        self.linear_out
            .forward(&out)
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
        build_rel_positional_embedding, parakeet_encoded_len_for_samples,
        parakeet_max_emittable_tokens, parakeet_preparation_bytes,
        parakeet_tensor_is_f16_projection, plan_encoder_windows, subsampled_len_3x,
        tdt_word_timestamps, EncoderGeometry, EncoderWindow, EncoderWindowing, Joint, LstmCell,
        ParakeetAsrModel, ParakeetDecoder, ParakeetEncoded, ParakeetEncoder, ParakeetLogMel,
        ParakeetNetwork, ParakeetPreparedEncoderArtifact, ParakeetPreprocessor,
        ParakeetRetainedDecodeBatchRow, ParakeetRetainedDecodeStep, ParakeetTimings, Predictor,
        TdtTokenFrame, DEFAULT_MAX_SYMBOLS, PRED_HIDDEN,
    };
    use super::{tdt_batch_labels_and_durations, tdt_label_and_duration};
    use crate::backends::{BackendKind, DTypeSelection};
    use crate::error::{Error, Result};
    use crate::model::ModelVariant;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{Linear, VarBuilder};

    use crate::kernels::f16_weight_linear::WeightOnlyF16Load;
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

    /// A NeMo-layout encoder checkpoint with random weights, loaded through
    /// the production loaders.
    fn random_encoder(geometry: EncoderGeometry, seed: u64, device: &Device) -> ParakeetEncoder {
        ParakeetEncoder::load(random_encoder_vb(geometry, seed, device), geometry, None).unwrap()
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
            x = layer.forward(&x, &pos).unwrap();
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
        encoder.encode_windows(
            feature_frames,
            encoded_len,
            windowing,
            window_features,
            &mut ParakeetTimings::default(),
            checkpoint,
        )
    }

    /// Largest absolute difference; NaN or infinity counts as infinitely far.
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
            let actual = conv.forward(&x).unwrap();
            assert!(crate::kernels::depthwise_conv::depthwise_conv1d_calls() > calls);

            let h = conv
                .pointwise_conv1
                .forward(&x.transpose(1, 2).unwrap())
                .unwrap();
            let gate = h.narrow(1, 0, TINY.dim).unwrap();
            let value = h.narrow(1, TINY.dim, TINY.dim).unwrap();
            let h = gate
                .broadcast_mul(&candle_nn::ops::sigmoid(&value).unwrap())
                .unwrap();
            let h = conv.depthwise_conv.forward(&h).unwrap();
            let h = conv.batch_norm.forward_t(&h, false).unwrap();
            let h = super::swish(&h).unwrap();
            let expected = conv
                .pointwise_conv2
                .forward(&h)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
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
            }];
            steps.extend(model.retained_decode_step_batch(&mut rows).unwrap());
        }
        let diagnostics = model
            .retained_decode_output(artifact, &state)
            .diagnostics
            .unwrap();
        (steps, diagnostics)
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

    /// Every projection the recipe lists is F16 and nothing else is.
    fn assert_projection_split(encoder: &ParakeetEncoder, f16: bool) {
        assert_eq!(encoder.pre_encode.out.as_f16().is_some(), f16);
        for layer in &encoder.layers {
            for projection in [
                &layer.ff1.linear1,
                &layer.ff1.linear2,
                &layer.ff2.linear1,
                &layer.ff2.linear2,
                &layer.self_attn.linear_q,
                &layer.self_attn.linear_k,
                &layer.self_attn.linear_v,
                &layer.self_attn.linear_out,
                &layer.self_attn.linear_pos,
            ] {
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
                [
                    &layer.ff1.linear1,
                    &layer.ff1.linear2,
                    &layer.ff2.linear1,
                    &layer.ff2.linear2,
                    &layer.self_attn.linear_q,
                    &layer.self_attn.linear_k,
                    &layer.self_attn.linear_v,
                    &layer.self_attn.linear_out,
                    &layer.self_attn.linear_pos,
                ]
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
        let other =
            ParakeetEncoder::load(random_encoder_vb(TINY, 35, &device), TINY, Some(&device))
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
            assert!(total(60 * minute, backend) <= 5 * GIB / 2, "{backend:?}");
        }

        // E = 1024 frames: the window term is final from here on and equals
        // report 11's 832 MiB on every backend (all its buffers are powers
        // of two at W = 1024).
        let full_window = parakeet_preparation_bytes(8192 * 160, BackendKind::Cpu)
            .unwrap()
            .window_bytes;
        assert_eq!(full_window, 832 << 20);
        for minutes in [2, 10, 30, 60] {
            for backend in backends {
                let bytes = parakeet_preparation_bytes(minutes * minute, backend).unwrap();
                assert_eq!(bytes.window_bytes, full_window);
            }
        }
        for samples in [16_000, 7 * minute + 4_321, 60 * minute] {
            assert!(total(samples, BackendKind::Metal) >= total(samples, BackendKind::Cpu));
        }
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
}
