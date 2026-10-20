//! Kokoro-82M native runtime integration scaffolding (Rust-only).
//!
//! This module intentionally isolates Kokoro-specific loading, phonemization,
//! voice-pack handling, and future Candle inference implementation from the
//! generic runtime orchestration layer.

pub(crate) mod admission;
mod albert;
mod config;
mod decoder;
pub(crate) mod phonemizer;
pub mod planning;
mod prosody;
mod text_encoder;
mod voice;

pub use config::KokoroConfig;
pub use prosody::kokoro_fused_lstm_steps;

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use candle_core::pickle::PthTensors;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use tracing::info;

use crate::backends::DeviceProfile;
use crate::error::{Error, Result};
use crate::models::shared::memory::accounting::TensorStorageAccounting;

use self::phonemizer::EspeakPhonemizer;
use self::prosody::{
    build_alignment_matrix, build_alignment_matrix_batch, KokoroProsodyBatchOutput,
    KokoroProsodyDebugOutput, KokoroProsodyOutput, KokoroProsodyPredictor,
};
use self::text_encoder::KokoroTextEncoder;
use self::voice::VoiceLibrary;
pub(crate) use self::voice::KOKORO_MODEL_MEMO_MAX_BYTES;

pub(crate) const CHECKPOINT_FILE: &str = "kokoro-v1_0.pth";
const CONFIG_FILE: &str = "config.json";
const VOICES_DIR: &str = "voices";
const KOKORO_MAX_PHONEMES_PER_INPUT_CHAR: usize = 16;
const KOKORO_MAX_PHONEMES_PER_CHUNK: usize = 510;
const KOKORO_MAX_CONTEXT_TOKENS: usize = KOKORO_MAX_PHONEMES_PER_CHUNK + 2;
const KOKORO_MAX_DURATION_BINS: usize = 50;
pub(super) const KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK: usize = 4_096;
const KOKORO_DIM_IN: usize = 64;
const KOKORO_HIDDEN_CHANNELS: usize = 512;
const KOKORO_MAX_CONV_CHANNELS: usize = 512;
const KOKORO_STYLE_CHANNELS: usize = 128;
const KOKORO_DURATION_LAYERS: usize = 3;
const KOKORO_MEL_CHANNELS: usize = 80;
const KOKORO_TOKEN_COUNT: usize = 178;
const KOKORO_TEXT_ENCODER_KERNEL: usize = 5;
const KOKORO_PLBERT_HIDDEN_CHANNELS: usize = 768;
const KOKORO_PLBERT_ATTENTION_HEADS: usize = 12;
const KOKORO_PLBERT_INTERMEDIATE_CHANNELS: usize = 2_048;
const KOKORO_PLBERT_LAYERS: usize = 12;
const KOKORO_DECODER_TIME_EXPANSION: usize = 2;
const KOKORO_GENERATOR_UPSAMPLE_INITIAL_CHANNELS: usize = 512;
const KOKORO_GENERATOR_UPSAMPLE_RATES: [usize; 2] = [10, 6];
const KOKORO_GENERATOR_UPSAMPLE_KERNELS: [usize; 2] = [20, 12];
const KOKORO_GENERATOR_ISTFT_HOP: usize = 5;
const KOKORO_GENERATOR_ISTFT_N_FFT: usize = 20;
const KOKORO_GENERATOR_RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const KOKORO_GENERATOR_RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
const KOKORO_GENERATOR_RESBLOCK_BRANCHES: usize = 3;
const KOKORO_MAX_SAMPLES_PER_DURATION_FRAME: usize = KOKORO_DECODER_TIME_EXPANSION
    * KOKORO_GENERATOR_UPSAMPLE_RATES[0]
    * KOKORO_GENERATOR_UPSAMPLE_RATES[1]
    * KOKORO_GENERATOR_ISTFT_HOP;
const KOKORO_INTER_CHUNK_PAUSE_SAMPLES: usize = 960;
const KOKORO_MIN_SPEED: f32 = 0.5;
const KOKORO_MAX_SPEED: f32 = 2.0;
const KOKORO_ACTIVATION_DTYPE_BYTES: u64 = std::mem::size_of::<f32>() as u64;
// CPU evaluates all three generator resblock variants concurrently: one shared
// input plus three live tensors per branch. Accelerators evaluate them in
// sequence; nine buffers cover the accumulator and unfused CUDA AdaIN/Snake
// intermediates without charging Metal/CUDA for CPU-only branch parallelism.
const KOKORO_CPU_LIVE_STAGE_BUFFERS: u64 = 1 + KOKORO_GENERATOR_RESBLOCK_BRANCHES as u64 * 3;
const KOKORO_ACCELERATOR_LIVE_STAGE_BUFFERS: u64 = 9;
// `synth_harmonic_source_kokoro` retains the upsampled F0, nine harmonics,
// voicing mask, phase/radian intermediates, and a collected sine buffer. Forty
// f32 values per final-rate sample is a conservative bound over those vectors.
const KOKORO_HARMONIC_HOST_F32_PER_SAMPLE: u64 = 40;
pub(crate) const CHECKPOINT_SUBMODULE_KEYS: &[&str] = &[
    "bert",
    "bert_encoder",
    "predictor",
    "text_encoder",
    "decoder",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KokoroOutputBudget {
    pub max_model_tokens: usize,
    pub max_chunks: usize,
    pub max_chunk_expanded_frames: usize,
    pub max_samples: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KokoroPeakWorkspace {
    pub host_bytes: u64,
    pub cpu_tensor_bytes: u64,
    pub accelerator_tensor_bytes: u64,
}

/// Whether `error` is the per-chunk expanded-frame cap, which splitting the
/// chunk's phonemes recovers from.
fn is_kokoro_frame_cap_error(error: &Error) -> bool {
    matches!(
        error,
        Error::InferenceError(message)
            if message.contains("hard frame contract") || message.contains("hard per-chunk contract")
    )
}

/// Synthesizes `phonemes` with `synth`; on the frame-cap error, splits at the
/// space nearest the middle (or the middle character) and synthesizes each
/// half the same way, concatenating the audio. Any other error is returned.
fn synthesize_splitting_on_frame_cap(
    phonemes: &str,
    synth: &mut dyn FnMut(&str) -> Result<Vec<f32>>,
) -> Result<Vec<f32>> {
    const MAX_DEPTH: usize = 12;
    fn go(
        phonemes: &str,
        synth: &mut dyn FnMut(&str) -> Result<Vec<f32>>,
        depth: usize,
    ) -> Result<Vec<f32>> {
        match synth(phonemes) {
            Err(error) if is_kokoro_frame_cap_error(&error) && depth < MAX_DEPTH => {
                let Some((left, right)) = split_phonemes_near_middle(phonemes) else {
                    return Err(error);
                };
                let mut samples = go(&left, synth, depth + 1)?;
                samples.extend(go(&right, synth, depth + 1)?);
                Ok(samples)
            }
            other => other,
        }
    }
    go(phonemes, synth, 0)
}

/// Two non-empty halves split at the space closest to the middle, or at the
/// middle character when the chunk has no inner space.
fn split_phonemes_near_middle(phonemes: &str) -> Option<(String, String)> {
    let chars = phonemes.trim().chars().collect::<Vec<_>>();
    if chars.len() < 2 {
        return None;
    }
    let middle = chars.len() / 2;
    let split = chars
        .iter()
        .enumerate()
        .filter(|(index, ch)| ch.is_whitespace() && *index > 0 && *index + 1 < chars.len())
        .map(|(index, _)| index)
        .min_by_key(|index| index.abs_diff(middle));
    let (left, right) = match split {
        Some(index) => (&chars[..index], &chars[index + 1..]),
        None => (&chars[..middle], &chars[middle..]),
    };
    let left = left.iter().collect::<String>().trim().to_string();
    let right = right.iter().collect::<String>().trim().to_string();
    (!left.is_empty() && !right.is_empty()).then_some((left, right))
}

pub(crate) fn kokoro_peak_workspace(max_chunk_expanded_frames: u64) -> Result<KokoroPeakWorkspace> {
    let decoder_time_expansion = u64::try_from(KOKORO_DECODER_TIME_EXPANSION)
        .map_err(|_| Error::Overloaded("Kokoro decoder time scale exceeds u64".to_string()))?;
    let generator_time_scale = KOKORO_GENERATOR_UPSAMPLE_RATES
        .iter()
        .try_fold(decoder_time_expansion, |scale, rate| {
            scale.checked_mul(u64::try_from(*rate).ok()?)
        })
        .ok_or_else(|| Error::Overloaded("Kokoro generator time scale overflowed".to_string()))?;
    let final_stage_channels = KOKORO_GENERATOR_UPSAMPLE_INITIAL_CHANNELS
        .checked_shr(KOKORO_GENERATOR_UPSAMPLE_RATES.len() as u32)
        .ok_or_else(|| {
            Error::Overloaded("Kokoro generator channel topology overflowed".to_string())
        })?;
    let final_stage_elements_per_frame = generator_time_scale
        .checked_mul(u64::try_from(final_stage_channels).map_err(|_| {
            Error::Overloaded("Kokoro generator channel count exceeds u64".to_string())
        })?)
        .ok_or_else(|| Error::Overloaded("Kokoro generator stage shape overflowed".to_string()))?;
    let harmonic_channels = u64::try_from(KOKORO_GENERATOR_ISTFT_N_FFT + 2)
        .map_err(|_| Error::Overloaded("Kokoro harmonic channels exceed u64".to_string()))?;
    let harmonic_elements_per_frame = generator_time_scale
        .checked_mul(harmonic_channels)
        .ok_or_else(|| Error::Overloaded("Kokoro harmonic shape overflowed".to_string()))?;
    // ASR [512,F] and the retained F0/N curves [1,2F] remain live while the
    // generator owns its stage tensors and harmonic features.
    let retained_elements_per_frame = u64::try_from(KOKORO_HIDDEN_CHANNELS)
        .ok()
        .and_then(|channels| channels.checked_add(decoder_time_expansion * 2))
        .ok_or_else(|| Error::Overloaded("Kokoro retained shape overflowed".to_string()))?;
    let tensor_bytes = |live_stage_buffers: u64, domain: &str| -> Result<u64> {
        let elements_per_frame = final_stage_elements_per_frame
            .checked_mul(live_stage_buffers)
            .and_then(|value| value.checked_add(harmonic_elements_per_frame))
            .and_then(|value| value.checked_add(retained_elements_per_frame))
            .ok_or_else(|| {
                Error::Overloaded(format!("Kokoro {domain} workspace shape overflowed"))
            })?;
        max_chunk_expanded_frames
            .checked_mul(elements_per_frame)
            .and_then(|value| value.checked_mul(KOKORO_ACTIVATION_DTYPE_BYTES))
            .ok_or_else(|| Error::Overloaded(format!("Kokoro {domain} workspace overflowed")))
    };
    let cpu_tensor_bytes = tensor_bytes(KOKORO_CPU_LIVE_STAGE_BUFFERS, "CPU tensor")?;
    let accelerator_tensor_bytes =
        tensor_bytes(KOKORO_ACCELERATOR_LIVE_STAGE_BUFFERS, "accelerator tensor")?;

    let final_rate_samples_per_frame = generator_time_scale
        .checked_mul(
            u64::try_from(KOKORO_GENERATOR_ISTFT_HOP)
                .map_err(|_| Error::Overloaded("Kokoro iSTFT hop exceeds u64".to_string()))?,
        )
        .ok_or_else(|| Error::Overloaded("Kokoro sample scale overflowed".to_string()))?;
    let host_bytes = max_chunk_expanded_frames
        .checked_mul(final_rate_samples_per_frame)
        .and_then(|value| value.checked_mul(KOKORO_HARMONIC_HOST_F32_PER_SAMPLE))
        .and_then(|value| value.checked_mul(KOKORO_ACTIVATION_DTYPE_BYTES))
        .ok_or_else(|| Error::Overloaded("Kokoro host workspace overflowed".to_string()))?;

    Ok(KokoroPeakWorkspace {
        host_bytes,
        cpu_tensor_bytes,
        accelerator_tensor_bytes,
    })
}

pub(crate) fn kokoro_output_budget(text: &str, speed: f32) -> Result<KokoroOutputBudget> {
    let text_chars = text.trim().chars().count();
    if text_chars == 0 {
        return Err(Error::InvalidInput(
            "Kokoro TTS input text is empty".to_string(),
        ));
    }
    let speed = normalize_kokoro_speed(speed)?;

    // Every successful chunk consumes at least one input character. The
    // phonemizer contract below limits each chunk to at most 16 IPA symbols per
    // input character, and the model adds one boundary token at each end.
    let max_chunks = text_chars;
    let max_phoneme_tokens = text_chars
        .checked_mul(KOKORO_MAX_PHONEMES_PER_INPUT_CHAR)
        .ok_or_else(|| Error::Overloaded("Kokoro phoneme budget overflowed".to_string()))?;
    let boundary_tokens = max_chunks
        .checked_mul(2)
        .ok_or_else(|| Error::Overloaded("Kokoro chunk-token budget overflowed".to_string()))?;
    let max_model_tokens = max_phoneme_tokens
        .checked_add(boundary_tokens)
        .ok_or_else(|| Error::Overloaded("Kokoro model-token budget overflowed".to_string()))?
        .min(
            max_chunks
                .checked_mul(KOKORO_MAX_CONTEXT_TOKENS)
                .ok_or_else(|| {
                    Error::Overloaded("Kokoro context-token budget overflowed".to_string())
                })?,
        );
    let max_duration_frames_per_token =
        ((KOKORO_MAX_DURATION_BINS as f64) / f64::from(speed)).ceil() as usize;
    let max_chunk_expanded_frames = max_model_tokens
        .min(KOKORO_MAX_CONTEXT_TOKENS)
        .checked_mul(max_duration_frames_per_token)
        .ok_or_else(|| {
            Error::Overloaded("Kokoro peak duration-frame budget overflowed".to_string())
        })?
        .min(KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK);
    let theoretical_total_frames = max_model_tokens
        .checked_mul(max_duration_frames_per_token)
        .ok_or_else(|| Error::Overloaded("Kokoro duration-frame budget overflowed".to_string()))?;
    let chunk_capped_total_frames = max_chunks
        .checked_mul(KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK)
        .ok_or_else(|| {
            Error::Overloaded("Kokoro chunked duration-frame budget overflowed".to_string())
        })?;
    let max_samples = theoretical_total_frames
        .min(chunk_capped_total_frames)
        .checked_mul(KOKORO_MAX_SAMPLES_PER_DURATION_FRAME)
        .and_then(|samples| {
            samples.checked_add(
                max_chunks
                    .saturating_sub(1)
                    .checked_mul(KOKORO_INTER_CHUNK_PAUSE_SAMPLES)?,
            )
        })
        .ok_or_else(|| Error::Overloaded("Kokoro output-sample budget overflowed".to_string()))?;

    Ok(KokoroOutputBudget {
        max_model_tokens,
        max_chunks,
        max_chunk_expanded_frames,
        max_samples,
    })
}

fn normalize_kokoro_speed(speed: f32) -> Result<f32> {
    if !speed.is_finite() {
        return Err(Error::InvalidInput(
            "Kokoro speed must be finite".to_string(),
        ));
    }
    Ok(speed.clamp(KOKORO_MIN_SPEED, KOKORO_MAX_SPEED))
}

fn kokoro_static_token_buckets(prepared: &[KokoroPreparedRequest]) -> Vec<Vec<usize>> {
    let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
    for (index, row) in prepared.iter().enumerate() {
        buckets.entry(row.token_ids.len()).or_default().push(index);
    }
    buckets.into_values().collect()
}

fn kokoro_duration_buckets(expanded_frames: &[usize]) -> Vec<Vec<usize>> {
    let mut buckets = BTreeMap::<usize, Vec<usize>>::new();
    for (index, &frames) in expanded_frames.iter().enumerate() {
        buckets.entry(frames).or_default().push(index);
    }
    buckets.into_values().collect()
}

fn kokoro_decoder_conditioning_frames(expanded_frames: usize) -> Result<usize> {
    expanded_frames.checked_mul(2).ok_or_else(|| {
        Error::Overloaded("Kokoro decoder conditioning length overflowed".to_string())
    })
}

fn select_trimmed_batch(tensor: &Tensor, rows: &[usize], frames: usize) -> Result<Tensor> {
    let selected = rows
        .iter()
        .map(|&row| {
            tensor
                .narrow(0, row, 1)
                .and_then(|tensor| tensor.narrow(2, 0, frames))
                .map_err(Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = selected.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Error::from)
}

fn select_trimmed_batch_2d(tensor: &Tensor, rows: &[usize], frames: usize) -> Result<Tensor> {
    let selected = rows
        .iter()
        .map(|&row| {
            tensor
                .narrow(0, row, 1)
                .and_then(|tensor| tensor.narrow(1, 0, frames))
                .map_err(Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = selected.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0).map_err(Error::from)
}

fn kokoro_samples_per_duration_frame(config: &KokoroConfig) -> Result<usize> {
    if config.istftnet.gen_istft_hop_size == 0
        || config.istftnet.upsample_rates.is_empty()
        || config.istftnet.upsample_rates.contains(&0)
    {
        return Err(Error::ModelLoadError(
            "Kokoro decoder sample scale must be nonzero".to_string(),
        ));
    }
    if config.istftnet.upsample_kernel_sizes.len() != config.istftnet.upsample_rates.len()
        || config
            .istftnet
            .upsample_kernel_sizes
            .iter()
            .zip(&config.istftnet.upsample_rates)
            .any(|(kernel, rate)| kernel < rate || (kernel - rate) % 2 != 0)
    {
        return Err(Error::ModelLoadError(
            "Kokoro decoder upsample kernels do not preserve the admitted sample scale".to_string(),
        ));
    }
    let generator_scale = config
        .istftnet
        .upsample_rates
        .iter()
        .try_fold(config.istftnet.gen_istft_hop_size, |scale, rate| {
            scale.checked_mul(*rate)
        })
        .ok_or_else(|| {
            Error::ModelLoadError("Kokoro decoder sample scale overflowed".to_string())
        })?;
    generator_scale
        .checked_mul(KOKORO_DECODER_TIME_EXPANSION)
        .ok_or_else(|| Error::ModelLoadError("Kokoro decoder sample scale overflowed".to_string()))
}

fn validate_kokoro_output_contract(config: &KokoroConfig) -> Result<()> {
    if config.context_length() == 0 || config.context_length() > KOKORO_MAX_CONTEXT_TOKENS {
        return Err(Error::ModelLoadError(format!(
            "Kokoro context length {} exceeds the admitted output contract ({KOKORO_MAX_CONTEXT_TOKENS})",
            config.context_length()
        )));
    }
    if config.max_dur == 0 || config.max_dur > KOKORO_MAX_DURATION_BINS {
        return Err(Error::ModelLoadError(format!(
            "Kokoro max_dur {} exceeds the admitted output contract ({KOKORO_MAX_DURATION_BINS})",
            config.max_dur
        )));
    }
    // The estimate above is for the one production Kokoro-82M topology loaded
    // by this runtime. Widths alter live tensors, while convolution kernels can
    // alter backend scratch even when padding preserves output length. Dropout
    // values and vocab contents are deliberately excluded: neither changes an
    // inference activation shape in this implementation.
    if config.dim_in != KOKORO_DIM_IN
        || config.hidden_dim != KOKORO_HIDDEN_CHANNELS
        || config.max_conv_dim != KOKORO_MAX_CONV_CHANNELS
        || config.style_dim != KOKORO_STYLE_CHANNELS
        || config.max_dur != KOKORO_MAX_DURATION_BINS
        || !config.multispeaker
        || config.n_layer != KOKORO_DURATION_LAYERS
        || config.n_mels != KOKORO_MEL_CHANNELS
        || config.n_token != KOKORO_TOKEN_COUNT
        || config.text_encoder_kernel_size != KOKORO_TEXT_ENCODER_KERNEL
        || config.plbert.hidden_size != KOKORO_PLBERT_HIDDEN_CHANNELS
        || config.plbert.num_attention_heads != KOKORO_PLBERT_ATTENTION_HEADS
        || config.plbert.intermediate_size != KOKORO_PLBERT_INTERMEDIATE_CHANNELS
        || config.plbert.max_position_embeddings != KOKORO_MAX_CONTEXT_TOKENS
        || config.plbert.num_hidden_layers != KOKORO_PLBERT_LAYERS
        || config.istftnet.upsample_initial_channel != KOKORO_GENERATOR_UPSAMPLE_INITIAL_CHANNELS
        || config.istftnet.upsample_rates.as_slice() != KOKORO_GENERATOR_UPSAMPLE_RATES.as_slice()
        || config.istftnet.upsample_kernel_sizes.as_slice()
            != KOKORO_GENERATOR_UPSAMPLE_KERNELS.as_slice()
        || config.istftnet.gen_istft_hop_size != KOKORO_GENERATOR_ISTFT_HOP
        || config.istftnet.gen_istft_n_fft != KOKORO_GENERATOR_ISTFT_N_FFT
        || config.istftnet.resblock_kernel_sizes.as_slice()
            != KOKORO_GENERATOR_RESBLOCK_KERNELS.as_slice()
        || config.istftnet.resblock_dilation_sizes.len()
            != config.istftnet.resblock_kernel_sizes.len()
        || config
            .istftnet
            .resblock_dilation_sizes
            .iter()
            .any(|dilations| dilations.as_slice() != KOKORO_GENERATOR_RESBLOCK_DILATIONS.as_slice())
    {
        return Err(Error::ModelLoadError(
            "Kokoro decoder topology does not match the admitted workspace contract".to_string(),
        ));
    }
    let sample_scale = kokoro_samples_per_duration_frame(config)?;
    if sample_scale > KOKORO_MAX_SAMPLES_PER_DURATION_FRAME {
        return Err(Error::ModelLoadError(format!(
            "Kokoro decoder sample scale {sample_scale} exceeds the admitted output contract ({KOKORO_MAX_SAMPLES_PER_DURATION_FRAME})"
        )));
    }
    Ok(())
}

fn kokoro_profile_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_KOKORO_PROFILE", false));
    ENABLED.get()
}

fn log_kokoro_profile(stage: &str, dur: Duration) {
    if kokoro_profile_enabled() {
        eprintln!(
            "kokoro profile: {stage} = {:.2} ms",
            dur.as_secs_f64() * 1_000.0
        );
    }
}

fn kokoro_cpu_predecoder_parallel_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_KOKORO_CPU_PREDECODER", true));
    ENABLED.get()
}

#[derive(Debug, Clone)]
pub struct KokoroPreparedRequest {
    pub source_text: String,
    pub requested_speaker: Option<String>,
    pub requested_language: Option<String>,
    pub requested_speed: f32,
    pub phonemes: String,
    pub token_ids: Vec<u32>,
    pub ref_style: Tensor,
    pub speed: f32,
}

impl KokoroPreparedRequest {
    pub(crate) fn retained_tensor_bytes(&self) -> Result<u64> {
        let mut accounting = TensorStorageAccounting::default();
        accounting.add_tensor(&self.ref_style).ok_or_else(|| {
            Error::Overloaded("Kokoro prepared style accounting overflowed".into())
        })?;
        Ok(accounting.bytes())
    }

    pub(crate) fn retained_host_bytes(&self) -> Result<u64> {
        let string_bytes = [
            self.source_text.capacity(),
            self.requested_speaker.as_ref().map_or(0, String::capacity),
            self.requested_language.as_ref().map_or(0, String::capacity),
            self.phonemes.capacity(),
        ]
        .into_iter()
        .try_fold(0usize, |bytes, capacity| bytes.checked_add(capacity))
        .ok_or_else(|| Error::Overloaded("Kokoro prepared host bytes overflowed".into()))?;
        let token_bytes = self
            .token_ids
            .capacity()
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::Overloaded("Kokoro prepared host bytes overflowed".into()))?;
        string_bytes
            .checked_add(token_bytes)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::Overloaded("Kokoro prepared host bytes overflowed".into()))
    }
}

#[derive(Debug, Clone)]
pub struct KokoroSynthesisResult {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub tokens_generated: usize,
    pub phonemes: String,
}

#[derive(Debug, Clone)]
pub struct KokoroPredecoderDebugOutput {
    pub prosody: KokoroProsodyDebugOutput,
    pub text_encoder_shape: Vec<usize>,
    pub asr_shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct KokoroPredecoderOutput {
    prosody: KokoroProsodyOutput,
    text_encoder_shape: Vec<usize>,
    asr: Tensor,
}

#[derive(Debug, Clone)]
struct KokoroPredecoderBatchOutput {
    prosody: KokoroProsodyBatchOutput,
    asr: Tensor,
}

#[derive(Debug)]
pub struct KokoroTtsModel {
    model_dir: PathBuf,
    checkpoint_path: PathBuf,
    config: KokoroConfig,
    device: DeviceProfile,
    dtype: DType,
    bert: albert::CustomAlbert,
    bert_encoder: Linear,
    prosody: KokoroProsodyPredictor,
    text_encoder: KokoroTextEncoder,
    decoder: decoder::KokoroDecoder,
    phonemizer: EspeakPhonemizer,
    voices: VoiceLibrary,
    checkpoint_tensor_counts: HashMap<String, usize>,
}

impl KokoroTtsModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join(CONFIG_FILE);
        let checkpoint_path = model_dir.join(CHECKPOINT_FILE);
        let voices_dir = model_dir.join(VOICES_DIR);

        if !config_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro config.json at {}",
                config_path.display()
            )));
        }
        if !checkpoint_path.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro checkpoint at {}",
                checkpoint_path.display()
            )));
        }
        if !voices_dir.exists() {
            return Err(Error::ModelNotFound(format!(
                "Missing Kokoro voices directory at {}",
                voices_dir.display()
            )));
        }

        let config: KokoroConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Kokoro config {}: {}",
                    config_path.display(),
                    e
                ))
            })?)?;
        validate_kokoro_output_contract(&config)?;

        let dtype = DType::F32;
        let checkpoint = KokoroCheckpoint::open(&checkpoint_path, &device.device, dtype)?;

        let phonemizer = EspeakPhonemizer::auto()?;
        let voices = VoiceLibrary::new(voices_dir, device.device.clone(), dtype)?;
        let bert = albert::CustomAlbert::load(
            &albert::AlbertModelConfig::from_kokoro(&config),
            checkpoint.submodule("bert")?.pp("module"),
        )?;
        let bert_encoder = candle_nn::linear(
            config.plbert.hidden_size,
            config.hidden_dim,
            checkpoint.submodule("bert_encoder")?.pp("module"),
        )
        .map_err(Error::from)?;
        let prosody = KokoroProsodyPredictor::load(&config, checkpoint.submodule("predictor")?)?;
        let text_encoder = KokoroTextEncoder::load(&config, checkpoint.submodule("text_encoder")?)?;
        let decoder = decoder::KokoroDecoder::load(&config, checkpoint.submodule("decoder")?)?;
        let checkpoint_tensor_counts = checkpoint.tensor_counts;

        info!(
            "Loaded Kokoro scaffolding from {:?} (phonemizer={}, submodules={:?})",
            model_dir,
            phonemizer.bin_path().display(),
            checkpoint_tensor_counts
        );

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            checkpoint_path,
            config,
            device,
            dtype,
            bert,
            bert_encoder,
            prosody,
            text_encoder,
            decoder,
            phonemizer,
            voices,
            checkpoint_tensor_counts,
        })
    }

    /// The voices Kokoro can speak. Japanese (`j*`) and Mandarin (`z*`)
    /// voice packs ship with the checkpoint but are hidden: their G2P is not
    /// ported, and requests for them are rejected as invalid input.
    pub fn available_speakers(&self) -> Result<Vec<String>> {
        Ok(self
            .voices
            .list_speakers()?
            .into_iter()
            .filter(|speaker| phonemizer::kokoro_speaker_supported(speaker))
            .collect())
    }

    /// Plan `text` into chunks within the voice pack's phoneme limit and
    /// `policy`'s character budgets, phonemizing the whole text once
    /// (KOK-11). Each chunk carries the phonemes [`Self::prepare_planned`]
    /// feeds the model, so nothing is phonemized again.
    pub fn plan_chunks(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        policy: planning::KokoroChunkPolicy,
    ) -> Result<Vec<planning::KokoroPlannedChunk>> {
        let speaker = self.resolve_speaker(speaker)?;
        let units = self
            .phonemizer
            .phonemize_units(text, language, Some(&speaker))?;
        Ok(planning::plan_chunks(
            units,
            policy,
            KOKORO_MAX_PHONEMES_PER_CHUNK,
        ))
    }

    /// The model inputs for a planned chunk, from its phonemes (no G2P).
    pub fn prepare_planned(
        &self,
        chunk: &planning::KokoroPlannedChunk,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest> {
        let requested_speaker = speaker.map(str::to_owned);
        let resolved = self.resolve_speaker(speaker)?;
        self.prepared_from_phonemes(
            &chunk.text,
            requested_speaker,
            language.map(str::to_owned),
            speed,
            &resolved,
            chunk.phonemes.clone(),
        )
    }

    pub fn prepare_request(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroPreparedRequest> {
        let requested_speaker = speaker.map(str::to_owned);
        let requested_language = language.map(str::to_owned);
        let requested_speed = speed;
        let speaker = self.resolve_speaker(speaker)?;
        let phonemes = self.phonemizer.phonemize(text, language, Some(&speaker))?;
        let phoneme_len = phonemes.chars().count();
        if phoneme_len == 0 {
            return Err(Error::InvalidInput(
                "Kokoro phonemizer produced no phonemes".to_string(),
            ));
        }
        let input_chars = text.trim().chars().count();
        let max_phonemes = input_chars
            .checked_mul(KOKORO_MAX_PHONEMES_PER_INPUT_CHAR)
            .ok_or_else(|| Error::InvalidInput("Kokoro input is too large".to_string()))?;
        if phoneme_len > max_phonemes {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme expansion {phoneme_len} exceeds the {KOKORO_MAX_PHONEMES_PER_INPUT_CHAR}-per-input-character contract ({max_phonemes})"
            )));
        }
        self.prepared_from_phonemes(
            text,
            requested_speaker,
            requested_language,
            requested_speed,
            &speaker,
            phonemes,
        )
    }

    /// The model inputs for an already phonemized chunk: token ids, the voice
    /// style row for its length, and the normalized speed.
    fn prepared_from_phonemes(
        &self,
        text: &str,
        requested_speaker: Option<String>,
        requested_language: Option<String>,
        requested_speed: f32,
        speaker: &str,
        phonemes: String,
    ) -> Result<KokoroPreparedRequest> {
        let phoneme_len = phonemes.chars().count();
        if phoneme_len == 0 {
            return Err(Error::InvalidInput(
                "Kokoro phonemizer produced no phonemes".to_string(),
            ));
        }
        if phoneme_len > KOKORO_MAX_PHONEMES_PER_CHUNK {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme sequence length {phoneme_len} exceeds supported voice-pack limit ({KOKORO_MAX_PHONEMES_PER_CHUNK}). Chunking is not implemented yet in the native runtime."
            )));
        }

        let token_ids = self.token_ids_from_phonemes(&phonemes)?;
        if token_ids.len() + 2 > self.config.context_length() {
            return Err(Error::InvalidInput(format!(
                "Kokoro phoneme token length {} exceeds context length {}",
                token_ids.len() + 2,
                self.config.context_length()
            )));
        }

        let ref_style = self.voices.style_for_phoneme_len(speaker, phoneme_len)?;
        let speed = normalize_kokoro_speed(requested_speed)?;

        Ok(KokoroPreparedRequest {
            source_text: text.to_string(),
            requested_speaker,
            requested_language,
            requested_speed,
            phonemes,
            token_ids,
            ref_style,
            speed,
        })
    }

    pub fn generate(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
    ) -> Result<KokoroSynthesisResult> {
        let t0 = Instant::now();
        let prepared = self.prepare_request(text, speaker, language, speed)?;
        log_kokoro_profile("tts.prepare_request", t0.elapsed());
        let mut results = self.generate_prepared_batch(&[prepared])?;
        results.pop().ok_or_else(|| {
            Error::InferenceError("Kokoro returned an empty scalar synthesis batch".to_string())
        })
    }

    /// Executes a ragged cohort as static token-width sub-batches followed by
    /// exact expanded-duration decoder buckets. Every multi-row bucket reaches
    /// ALBERT/text/prosody or the decoder as one native `B > 1` tensor call.
    pub fn generate_prepared_batch(
        &self,
        prepared: &[KokoroPreparedRequest],
    ) -> Result<Vec<KokoroSynthesisResult>> {
        match self.generate_prepared_batch_once(prepared) {
            Err(error) if is_kokoro_frame_cap_error(&error) => {
                // A row whose predicted durations overflow the per-chunk
                // frame cap (slow speech, long pauses) would fail the whole
                // cohort. Rerun each row alone, splitting only rows that
                // overflow (KOK-13).
                prepared
                    .iter()
                    .map(|row| self.generate_row_splitting_on_frame_cap(row))
                    .collect()
            }
            other => other,
        }
    }

    /// One row, split at a word boundary near the middle of its phonemes
    /// (recursively) while its predicted durations overflow the frame cap;
    /// the halves' audio is concatenated in order.
    fn generate_row_splitting_on_frame_cap(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroSynthesisResult> {
        let speaker = self.resolve_speaker(prepared.requested_speaker.as_deref())?;
        let mut tokens_generated = 0usize;
        let samples = synthesize_splitting_on_frame_cap(&prepared.phonemes, &mut |phonemes| {
            let chunk = if phonemes == prepared.phonemes {
                prepared.clone()
            } else {
                self.prepared_from_phonemes(
                    &prepared.source_text,
                    prepared.requested_speaker.clone(),
                    prepared.requested_language.clone(),
                    prepared.requested_speed,
                    &speaker,
                    phonemes.to_string(),
                )?
            };
            let mut results = self.generate_prepared_batch_once(std::slice::from_ref(&chunk))?;
            let result = results.pop().ok_or_else(|| {
                Error::InferenceError("Kokoro returned an empty scalar synthesis batch".into())
            })?;
            tokens_generated += result.tokens_generated;
            Ok(result.samples)
        })?;
        Ok(KokoroSynthesisResult {
            samples,
            sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
            tokens_generated,
            phonemes: prepared.phonemes.clone(),
        })
    }

    fn generate_prepared_batch_once(
        &self,
        prepared: &[KokoroPreparedRequest],
    ) -> Result<Vec<KokoroSynthesisResult>> {
        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let max_batch_size = self.max_native_batch_size();
        if prepared.len() > max_batch_size {
            return Err(Error::Overloaded(format!(
                "Kokoro native batch width {} exceeds this backend's static limit {max_batch_size}",
                prepared.len()
            )));
        }
        let mut results = vec![None; prepared.len()];
        let token_buckets = kokoro_static_token_buckets(prepared);
        for bucket in token_buckets {
            self.generate_token_bucket(prepared, &bucket, &mut results)?;
        }
        results
            .into_iter()
            .enumerate()
            .map(|(row, result)| {
                result.ok_or_else(|| {
                    Error::InferenceError(format!(
                        "Kokoro synthesis batch did not produce row {row}"
                    ))
                })
            })
            .collect()
    }

    pub fn max_native_batch_size(&self) -> usize {
        self.device.capabilities.recommended_batch_size.max(1)
    }

    fn generate_token_bucket(
        &self,
        prepared: &[KokoroPreparedRequest],
        indices: &[usize],
        results: &mut [Option<KokoroSynthesisResult>],
    ) -> Result<()> {
        let rows = indices
            .iter()
            .map(|&index| &prepared[index])
            .collect::<Vec<_>>();
        let t1 = Instant::now();
        let predecoder = self.run_predecoder_batch(&rows)?;
        log_kokoro_profile("tts.predecoder", t1.elapsed());
        let duration_buckets = kokoro_duration_buckets(&predecoder.prosody.expanded_frames);
        for duration_bucket in duration_buckets {
            let expanded_frames = predecoder.prosody.expanded_frames[duration_bucket[0]];
            let conditioning_frames = kokoro_decoder_conditioning_frames(expanded_frames)?;
            let asr = select_trimmed_batch(&predecoder.asr, &duration_bucket, expanded_frames)?;
            let f0 = select_trimmed_batch_2d(
                &predecoder.prosody.f0,
                &duration_bucket,
                conditioning_frames,
            )?;
            let n = select_trimmed_batch_2d(
                &predecoder.prosody.n,
                &duration_bucket,
                conditioning_frames,
            )?;
            let style_rows = duration_bucket
                .iter()
                .map(|&row| {
                    rows[row]
                        .ref_style
                        .i((.., 0..self.config.style_dim))
                        .map_err(Error::from)
                })
                .collect::<Result<Vec<_>>>()?;
            let style_refs = style_rows.iter().collect::<Vec<_>>();
            let style = Tensor::cat(&style_refs, 0).map_err(Error::from)?;
            let seeds = vec![None; duration_bucket.len()];
            let t2 = Instant::now();
            let samples = self.decoder.forward_batch(&asr, &f0, &n, &style, &seeds)?;
            log_kokoro_profile("tts.decoder", t2.elapsed());
            for (&bucket_row, samples) in duration_bucket.iter().zip(samples) {
                let request_index = indices[bucket_row];
                let max_samples = self.output_sample_limit(expanded_frames)?;
                if samples.len() > max_samples || samples.capacity() > max_samples {
                    return Err(Error::InferenceError(format!(
                        "Kokoro decoder output exceeded its hard sample contract: len={}, capacity={}, max={max_samples}",
                        samples.len(),
                        samples.capacity()
                    )));
                }
                results[request_index] = Some(KokoroSynthesisResult {
                    tokens_generated: prepared[request_index].token_ids.len(),
                    phonemes: prepared[request_index].phonemes.clone(),
                    sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
                    samples,
                });
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn generate_with_seed_for_test(
        &self,
        text: &str,
        speaker: Option<&str>,
        language: Option<&str>,
        speed: f32,
        rng_seed: u64,
    ) -> Result<KokoroSynthesisResult> {
        let t0 = Instant::now();
        let prepared = self.prepare_request(text, speaker, language, speed)?;
        log_kokoro_profile("tts.prepare_request", t0.elapsed());
        let t1 = Instant::now();
        let predecoder = self.run_predecoder(&prepared)?;
        log_kokoro_profile("tts.predecoder", t1.elapsed());
        let style = prepared
            .ref_style
            .i((.., 0..self.config.style_dim))
            .map_err(Error::from)?;
        let max_samples = self.output_sample_limit(predecoder.prosody.expanded_frames)?;
        let t2 = Instant::now();
        let samples = self.decoder.forward_with_seed(
            &predecoder.asr,
            &predecoder.prosody.f0,
            &predecoder.prosody.n,
            &style,
            Some(rng_seed),
        )?;
        if samples.len() > max_samples || samples.capacity() > max_samples {
            return Err(Error::InferenceError(format!(
                "Kokoro decoder output exceeded its hard sample contract: len={}, capacity={}, max={max_samples}",
                samples.len(),
                samples.capacity()
            )));
        }
        log_kokoro_profile("tts.decoder", t2.elapsed());
        log_kokoro_profile("tts.total", t0.elapsed());
        Ok(KokoroSynthesisResult {
            tokens_generated: prepared.token_ids.len(),
            phonemes: prepared.phonemes,
            sample_rate: KokoroConfig::TARGET_SAMPLE_RATE,
            samples,
        })
    }

    pub fn config(&self) -> &KokoroConfig {
        &self.config
    }

    fn output_sample_limit(&self, expanded_frames: usize) -> Result<usize> {
        if expanded_frames > KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK {
            return Err(Error::InferenceError(format!(
                "Kokoro expanded-frame count exceeded its hard per-chunk contract: {expanded_frames} > {KOKORO_MAX_EXPANDED_FRAMES_PER_CHUNK}"
            )));
        }
        expanded_frames
            .checked_mul(kokoro_samples_per_duration_frame(&self.config)?)
            .ok_or_else(|| Error::Overloaded("Kokoro output-sample limit overflowed".to_string()))
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint_path
    }

    pub fn checkpoint_tensor_counts(&self) -> &HashMap<String, usize> {
        &self.checkpoint_tensor_counts
    }

    pub fn run_bert_prosody_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyDebugOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        self.run_bert_prosody_debug_for_input(&input_ids, prepared)
    }

    fn run_bert_prosody(&self, prepared: &KokoroPreparedRequest) -> Result<KokoroProsodyOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        self.run_bert_prosody_for_input(&input_ids, prepared)
    }

    fn run_bert_prosody_debug_for_input(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyDebugOutput> {
        let bert_hidden = self.bert.forward(input_ids, None)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody
            .forward_debug(&d_en, &prepared.ref_style, prepared.speed)
    }

    fn run_bert_prosody_for_input(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroProsodyOutput> {
        let bert_hidden = self.bert.forward(input_ids, None)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        self.prosody
            .forward(&d_en, &prepared.ref_style, prepared.speed)
    }

    pub fn run_predecoder_debug(
        &self,
        prepared: &KokoroPreparedRequest,
    ) -> Result<KokoroPredecoderDebugOutput> {
        let out = self.run_predecoder(prepared)?;
        Ok(KokoroPredecoderDebugOutput {
            prosody: KokoroProsodyDebugOutput {
                duration_frames: out.prosody.duration_frames.clone(),
                expanded_frames: out.prosody.expanded_frames,
                f0_shape: out.prosody.f0.shape().dims().to_vec(),
                n_shape: out.prosody.n.shape().dims().to_vec(),
            },
            text_encoder_shape: out.text_encoder_shape,
            asr_shape: out.asr.shape().dims().to_vec(),
        })
    }

    fn run_predecoder(&self, prepared: &KokoroPreparedRequest) -> Result<KokoroPredecoderOutput> {
        let input_ids = self.build_model_input_ids(prepared)?;
        let (prosody, t_en) = self.run_predecoder_branches(&input_ids, prepared)?;
        let pred_aln = build_alignment_matrix(&prosody.duration_frames, &self.device.device)?;
        let asr = t_en
            .contiguous()
            .map_err(Error::from)?
            .matmul(&pred_aln.contiguous().map_err(Error::from)?)
            .map_err(Error::from)?;
        let text_encoder_shape = t_en.shape().dims().to_vec();
        Ok(KokoroPredecoderOutput {
            prosody,
            text_encoder_shape,
            asr,
        })
    }

    fn run_predecoder_batch(
        &self,
        prepared: &[&KokoroPreparedRequest],
    ) -> Result<KokoroPredecoderBatchOutput> {
        let Some(first) = prepared.first() else {
            return Err(Error::InvalidInput(
                "Kokoro predecoder batch cannot be empty".to_string(),
            ));
        };
        if prepared.len() == 1 {
            let scalar = self.run_predecoder(first)?;
            return Ok(KokoroPredecoderBatchOutput {
                prosody: KokoroProsodyBatchOutput {
                    duration_frames: vec![scalar.prosody.duration_frames],
                    expanded_frames: vec![scalar.prosody.expanded_frames],
                    f0: scalar.prosody.f0,
                    n: scalar.prosody.n,
                },
                asr: scalar.asr,
            });
        }
        let token_width = first.token_ids.len();
        if prepared
            .iter()
            .any(|row| row.token_ids.len() != token_width)
        {
            return Err(Error::InvalidInput(
                "Kokoro static predecoder batch requires equal token widths".to_string(),
            ));
        }
        let input_rows = prepared
            .iter()
            .map(|row| self.build_model_input_ids(row))
            .collect::<Result<Vec<_>>>()?;
        let input_refs = input_rows.iter().collect::<Vec<_>>();
        let input_ids = Tensor::cat(&input_refs, 0).map_err(Error::from)?;
        let style_refs = prepared
            .iter()
            .map(|row| &row.ref_style)
            .collect::<Vec<_>>();
        let ref_style = Tensor::cat(&style_refs, 0).map_err(Error::from)?;
        let speeds = prepared.iter().map(|row| row.speed).collect::<Vec<_>>();

        let bert_hidden = self.bert.forward(&input_ids, None)?;
        let d_en = self
            .bert_encoder
            .forward(&bert_hidden)
            .map_err(Error::from)?
            .transpose(1, 2)
            .map_err(Error::from)?;
        let prosody = self.prosody.forward_batch(&d_en, &ref_style, &speeds)?;
        let t_en = self.text_encoder.forward(&input_ids)?;
        let pred_aln = build_alignment_matrix_batch(&prosody.duration_frames, &self.device.device)?;
        let asr = t_en
            .contiguous()
            .map_err(Error::from)?
            .matmul(&pred_aln.contiguous().map_err(Error::from)?)
            .map_err(Error::from)?;
        Ok(KokoroPredecoderBatchOutput { prosody, asr })
    }

    fn run_predecoder_branches(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<(KokoroProsodyOutput, Tensor)> {
        if input_ids.device().is_cpu() && kokoro_cpu_predecoder_parallel_enabled() {
            return self.run_predecoder_branches_parallel_cpu(input_ids, prepared);
        }

        let prosody = self.run_bert_prosody_for_input(input_ids, prepared)?;
        let t_en = self.text_encoder.forward(input_ids)?;
        Ok((prosody, t_en))
    }

    fn run_predecoder_branches_parallel_cpu(
        &self,
        input_ids: &Tensor,
        prepared: &KokoroPreparedRequest,
    ) -> Result<(KokoroProsodyOutput, Tensor)> {
        thread::scope(|scope| {
            let prosody_handle = scope.spawn(|| {
                self.run_bert_prosody_for_input(input_ids, prepared)
                    .map_err(|e| e.to_string())
            });
            let text_handle = scope.spawn(|| {
                self.text_encoder
                    .forward(input_ids)
                    .map_err(|e| e.to_string())
            });

            let prosody = match prosody_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro predecoder prosody worker thread panicked".to_string(),
                    ))
                }
            };
            let t_en = match text_handle.join() {
                Ok(Ok(t)) => t,
                Ok(Err(msg)) => return Err(Error::InferenceError(msg)),
                Err(_) => {
                    return Err(Error::InferenceError(
                        "Kokoro predecoder text encoder worker thread panicked".to_string(),
                    ))
                }
            };

            Ok((prosody, t_en))
        })
    }

    fn build_model_input_ids(&self, prepared: &KokoroPreparedRequest) -> Result<Tensor> {
        let mut input_ids = Vec::with_capacity(prepared.token_ids.len() + 2);
        input_ids.push(0u32);
        input_ids.extend_from_slice(&prepared.token_ids);
        input_ids.push(0u32);
        let seq_len = input_ids.len();
        Tensor::from_vec(input_ids, (1, seq_len), &self.device.device).map_err(Error::from)
    }

    fn resolve_speaker(&self, requested: Option<&str>) -> Result<String> {
        let speakers = self.available_speakers()?;
        if speakers.is_empty() {
            return Err(Error::ModelLoadError(
                "Kokoro voices directory is empty".to_string(),
            ));
        }
        let requested = requested
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("af_heart");
        if !phonemizer::kokoro_speaker_supported(requested) {
            // Japanese and Mandarin voices: the typed unsupported-language error.
            phonemizer::kokoro_voice_for(None, Some(requested))?;
        }
        if let Some(exact) = speakers.iter().find(|s| s.as_str() == requested) {
            return Ok(exact.clone());
        }
        let requested_lower = requested.to_ascii_lowercase();
        if let Some(casefold) = speakers
            .iter()
            .find(|s| s.to_ascii_lowercase() == requested_lower)
        {
            return Ok(casefold.clone());
        }
        if let Some(alias) = openai_voice_alias(&requested_lower) {
            if let Some(voice) = speakers.iter().find(|s| s.as_str() == alias) {
                return Ok(voice.clone());
            }
        }
        Err(Error::InvalidInput(format!(
            "Unknown Kokoro speaker '{requested}'. Available speakers: {}",
            speakers.join(", ")
        )))
    }

    fn token_ids_from_phonemes(&self, phonemes: &str) -> Result<Vec<u32>> {
        let mut token_ids = Vec::with_capacity(phonemes.chars().count());
        let mut unknown = Vec::new();
        for ch in phonemes.chars() {
            let key = ch.to_string();
            if let Some(id) = self.config.vocab.get(&key) {
                token_ids.push(*id);
            } else if ch.is_whitespace() {
                if let Some(id) = self.config.vocab.get(" ") {
                    token_ids.push(*id);
                }
            } else {
                unknown.push(ch);
            }
        }

        if token_ids.is_empty() {
            return Err(Error::TokenizationError(format!(
                "Kokoro phoneme tokenizer produced zero tokens (unknown chars: {:?})",
                unknown
            )));
        }

        if !unknown.is_empty() {
            tracing::warn!(
                "Kokoro phoneme tokenizer skipped {} unknown symbols: {:?}",
                unknown.len(),
                unknown
            );
        }

        Ok(token_ids)
    }
}

/// The Kokoro voice for an OpenAI voice name, for clients that send
/// OpenAI's (`alloy`, `nova`…). Kokoro has five of them; the rest get a
/// similar voice.
fn openai_voice_alias(name: &str) -> Option<&'static str> {
    Some(match name {
        "alloy" => "af_alloy",
        "echo" => "am_echo",
        "fable" => "bm_fable",
        "nova" => "af_nova",
        "onyx" => "am_onyx",
        "ash" => "am_adam",
        "ballad" => "bm_george",
        "cedar" => "am_liam",
        "coral" => "af_heart",
        "marin" => "af_river",
        "sage" => "af_sarah",
        "shimmer" => "af_bella",
        "verse" => "am_michael",
        _ => return None,
    })
}

#[cfg(test)]
#[test]
fn openai_voice_names_map_to_kokoro_voices() {
    assert_eq!(openai_voice_alias("alloy"), Some("af_alloy"));
    assert_eq!(openai_voice_alias("shimmer"), Some("af_bella"));
    assert_eq!(openai_voice_alias("af_heart"), None);
    // Every alias names a Kokoro voice.
    for name in [
        "alloy", "echo", "fable", "nova", "onyx", "ash", "ballad", "cedar", "coral", "marin",
        "sage", "shimmer", "verse",
    ] {
        let voice = openai_voice_alias(name).unwrap();
        assert!(phonemizer::kokoro_speaker_supported(voice), "{voice}");
    }
}

static KOKORO_CHECKPOINT_PARSES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static THREAD_CHECKPOINT_PARSES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Process-wide count of Kokoro checkpoint state-dict parses (one pickle
/// read of `kokoro-v1_0.pth` per submodule).
pub fn kokoro_checkpoint_parses() -> u64 {
    KOKORO_CHECKPOINT_PARSES.load(Ordering::Relaxed)
}

#[cfg(test)]
fn thread_checkpoint_parses() -> u64 {
    THREAD_CHECKPOINT_PARSES.with(|parses| parses.get())
}

fn record_checkpoint_parse() {
    KOKORO_CHECKPOINT_PARSES.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_CHECKPOINT_PARSES.with(|parses| parses.set(parses.get() + 1));
}

/// The checkpoint's submodule state dicts, each parsed once and shared by
/// validation (the tensor counts) and weight loading (the VarBuilders).
struct KokoroCheckpoint {
    submodules: HashMap<&'static str, VarBuilder<'static>>,
    tensor_counts: HashMap<String, usize>,
}

impl KokoroCheckpoint {
    fn open(checkpoint_path: &Path, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let mut submodules = HashMap::new();
        let mut tensor_counts = HashMap::new();
        for key in CHECKPOINT_SUBMODULE_KEYS {
            let tensors = PthTensors::new(checkpoint_path, Some(key)).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to inspect Kokoro checkpoint submodule '{key}' in {}: {}",
                    checkpoint_path.display(),
                    e
                ))
            })?;
            record_checkpoint_parse();
            let count = tensors.tensor_infos().len();
            if count == 0 {
                return Err(Error::ModelLoadError(format!(
                    "Kokoro checkpoint submodule '{key}' in {} has no tensors",
                    checkpoint_path.display()
                )));
            }
            submodules.insert(
                *key,
                VarBuilder::from_backend(Box::new(tensors), dtype, device.clone()),
            );
            tensor_counts.insert((*key).to_string(), count);
        }
        Ok(Self {
            submodules,
            tensor_counts,
        })
    }

    fn submodule(&self, key: &str) -> Result<VarBuilder<'static>> {
        self.submodules.get(key).cloned().ok_or_else(|| {
            Error::ModelLoadError(format!(
                "Kokoro checkpoint submodule '{key}' was not opened"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{DeviceKind, DeviceSelector};
    use rustfft::num_complex::Complex32;
    use rustfft::FftPlanner;
    use std::path::Path;

    /// KOK-13: the real frame-cap error is recognized as recoverable.
    #[test]
    fn duration_frame_cap_error_is_recognized_for_splitting() {
        let error = prosody::bounded_duration_frames(vec![100.0; 64], 4_096).unwrap_err();
        assert!(is_kokoro_frame_cap_error(&error), "{error}");
        assert!(!is_kokoro_frame_cap_error(&Error::InferenceError(
            "Kokoro duration predictor produced a non-finite value".into()
        )));
    }

    /// KOK-13: a chunk whose durations overflow the frame cap is split at the
    /// space nearest its middle, recursively, and the halves' audio is
    /// concatenated in order; other errors are not retried.
    #[test]
    fn frame_cap_overflow_splits_the_chunk_and_keeps_order() {
        assert_eq!(
            split_phonemes_near_middle("ab cd ef"),
            Some(("ab cd".to_string(), "ef".to_string()))
        );
        assert_eq!(
            split_phonemes_near_middle("abcdef"),
            Some(("abc".to_string(), "def".to_string()))
        );
        assert_eq!(split_phonemes_near_middle("a"), None);

        let cap_error = || Error::InferenceError("exceeded its hard frame contract".into());
        let mut calls = Vec::new();
        let audio = synthesize_splitting_on_frame_cap(
            "hˈɛ wˈɜː lˈd ɪz bɪɡ ænd ɹˈaʊnd",
            &mut |chunk| {
                calls.push(chunk.to_string());
                if chunk.chars().count() > 8 {
                    return Err(cap_error());
                }
                Ok(chunk
                    .chars()
                    .filter(|ch| !ch.is_whitespace())
                    .map(|ch| ch as u32 as f32)
                    .collect())
            },
        )
        .unwrap();
        let expected = "hˈɛwˈɜːlˈdɪzbɪɡændɹˈaʊnd"
            .chars()
            .map(|ch| ch as u32 as f32)
            .collect::<Vec<_>>();
        assert_eq!(audio, expected);
        assert!(calls.len() > 3);

        let mut other =
            |_: &str| -> Result<Vec<f32>> { Err(Error::InferenceError("unrelated".into())) };
        assert!(synthesize_splitting_on_frame_cap("a b c d", &mut other).is_err());
    }

    fn prepared_with_width(width: usize) -> KokoroPreparedRequest {
        KokoroPreparedRequest {
            source_text: "a".repeat(width),
            requested_speaker: None,
            requested_language: None,
            requested_speed: 1.0,
            phonemes: "a".repeat(width),
            token_ids: vec![1; width],
            ref_style: Tensor::zeros((1, 256), DType::F32, &candle_core::Device::Cpu)
                .expect("style"),
            speed: 1.0,
        }
    }

    #[test]
    fn prepared_request_accounts_host_and_tensor_ownership_separately() {
        let prepared = prepared_with_width(7);
        assert_eq!(prepared.retained_tensor_bytes().unwrap(), 256 * 4);
        assert!(prepared.retained_host_bytes().unwrap() >= 7 * (2 + 4));
    }

    #[test]
    fn ragged_requests_are_bucketed_by_static_token_width_stably() {
        let prepared = vec![
            prepared_with_width(7),
            prepared_with_width(3),
            prepared_with_width(7),
            prepared_with_width(5),
            prepared_with_width(3),
        ];
        assert_eq!(
            kokoro_static_token_buckets(&prepared),
            vec![vec![1, 4], vec![3], vec![0, 2]]
        );
    }

    #[test]
    fn decoder_rows_are_bucketed_by_exact_predicted_duration_stably() {
        assert_eq!(
            kokoro_duration_buckets(&[11, 7, 11, 9, 7]),
            vec![vec![1, 4], vec![3], vec![0, 2]]
        );
    }

    #[test]
    fn decoder_conditioning_preserves_the_double_rate_f0_and_noise_curves() {
        assert_eq!(kokoro_decoder_conditioning_frames(88).unwrap(), 176);
        assert!(matches!(
            kokoro_decoder_conditioning_frames(usize::MAX),
            Err(Error::Overloaded(message)) if message.contains("conditioning length")
        ));
    }

    #[test]
    fn ragged_alignment_batch_pads_only_the_frame_axis() {
        let alignment = build_alignment_matrix_batch(
            &[vec![1, 2, 1], vec![2, 2, 2]],
            &candle_core::Device::Cpu,
        )
        .expect("alignment");
        assert_eq!(alignment.shape().dims(), &[2, 3, 6]);
        let rows = alignment.to_vec3::<f32>().expect("alignment values");
        assert_eq!(rows[0][0], vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(rows[0][1], vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        assert_eq!(rows[0][2], vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    fn canonical_config() -> KokoroConfig {
        KokoroConfig {
            istftnet: config::KokoroIstftNetConfig {
                upsample_kernel_sizes: vec![20, 12],
                upsample_rates: vec![10, 6],
                gen_istft_hop_size: 5,
                gen_istft_n_fft: 20,
                resblock_dilation_sizes: vec![vec![1, 3, 5]; 3],
                resblock_kernel_sizes: vec![3, 7, 11],
                upsample_initial_channel: 512,
            },
            dim_in: 64,
            dropout: 0.2,
            hidden_dim: 512,
            max_conv_dim: 512,
            max_dur: 50,
            multispeaker: true,
            n_layer: 3,
            n_mels: 80,
            n_token: 178,
            style_dim: 128,
            text_encoder_kernel_size: 5,
            plbert: config::KokoroPlbertConfig {
                hidden_size: 768,
                num_attention_heads: 12,
                intermediate_size: 2048,
                max_position_embeddings: 512,
                num_hidden_layers: 12,
                dropout: 0.1,
            },
            vocab: HashMap::new(),
        }
    }

    #[test]
    fn kokoro_config_context_length_uses_plbert_positions() {
        let cfg = canonical_config();

        assert_eq!(cfg.context_length(), 512);
        assert_eq!(kokoro_samples_per_duration_frame(&cfg).unwrap(), 600);
        validate_kokoro_output_contract(&cfg).unwrap();
    }

    #[test]
    fn kokoro_workspace_contract_rejects_unadmitted_decoder_topology() {
        let mut cfg = canonical_config();
        cfg.istftnet.upsample_rates = vec![8, 8];

        assert!(matches!(
            validate_kokoro_output_contract(&cfg),
            Err(Error::ModelLoadError(message)) if message.contains("workspace contract")
        ));
    }

    #[test]
    fn kokoro_workspace_contract_rejects_every_shape_bearing_topology_change() {
        let mutations: &[(&str, fn(&mut KokoroConfig))] = &[
            ("dim_in", |cfg| cfg.dim_in += 1),
            ("hidden_dim", |cfg| cfg.hidden_dim += 1),
            ("max_conv_dim", |cfg| cfg.max_conv_dim += 1),
            ("max_dur", |cfg| cfg.max_dur -= 1),
            ("multispeaker", |cfg| cfg.multispeaker = false),
            ("n_layer", |cfg| cfg.n_layer += 1),
            ("n_mels", |cfg| cfg.n_mels += 1),
            ("n_token", |cfg| cfg.n_token += 1),
            ("style_dim", |cfg| cfg.style_dim += 1),
            ("text_encoder_kernel_size", |cfg| {
                cfg.text_encoder_kernel_size += 2
            }),
            ("plbert.hidden_size", |cfg| cfg.plbert.hidden_size += 1),
            ("plbert.num_attention_heads", |cfg| {
                cfg.plbert.num_attention_heads -= 1
            }),
            ("plbert.intermediate_size", |cfg| {
                cfg.plbert.intermediate_size += 1
            }),
            ("plbert.max_position_embeddings", |cfg| {
                cfg.plbert.max_position_embeddings -= 1
            }),
            ("plbert.num_hidden_layers", |cfg| {
                cfg.plbert.num_hidden_layers += 1
            }),
            ("upsample_initial_channel", |cfg| {
                cfg.istftnet.upsample_initial_channel += 1
            }),
            ("upsample_rates", |cfg| cfg.istftnet.upsample_rates[0] = 8),
            ("upsample_kernel_sizes", |cfg| {
                cfg.istftnet.upsample_kernel_sizes[0] += 2
            }),
            ("gen_istft_hop_size", |cfg| {
                cfg.istftnet.gen_istft_hop_size += 1
            }),
            ("gen_istft_n_fft", |cfg| cfg.istftnet.gen_istft_n_fft += 2),
            ("resblock_kernel_sizes", |cfg| {
                cfg.istftnet.resblock_kernel_sizes[1] += 2
            }),
            ("resblock_dilation_sizes", |cfg| {
                cfg.istftnet.resblock_dilation_sizes[0][1] += 1
            }),
        ];

        for (field, mutate) in mutations {
            let mut cfg = canonical_config();
            mutate(&mut cfg);
            assert!(
                matches!(
                    validate_kokoro_output_contract(&cfg),
                    Err(Error::ModelLoadError(message)) if message.contains("workspace contract")
                ),
                "topology mutation {field} was accepted"
            );
        }
    }

    #[test]
    fn kokoro_peak_workspace_arithmetic_is_checked() {
        assert!(matches!(
            kokoro_peak_workspace(u64::MAX),
            Err(Error::Overloaded(message)) if message.contains("workspace overflowed")
        ));
    }

    /// KOK-19: loading parses each checkpoint submodule's pickle once (five
    /// parses), sharing it between validation and the weight VarBuilders;
    /// it used to be parsed three times per submodule (fifteen).
    #[test]
    #[ignore = "requires the local Kokoro artifact (IZWI_KOKORO_MODEL_DIR)"]
    fn kokoro_load_parses_each_checkpoint_submodule_once() {
        let model_dir = {
            let _guard = crate::env_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            std::env::var_os("IZWI_KOKORO_MODEL_DIR")
                .expect("set IZWI_KOKORO_MODEL_DIR to the local Kokoro-82M directory")
        };
        let device = DeviceSelector::detect_with_preference(Some("cpu")).expect("cpu device");
        let before = thread_checkpoint_parses();
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        assert_eq!(
            thread_checkpoint_parses() - before,
            CHECKPOINT_SUBMODULE_KEYS.len() as u64
        );
        let counts = model.checkpoint_tensor_counts();
        assert_eq!(counts.len(), CHECKPOINT_SUBMODULE_KEYS.len());
        assert!(counts.values().all(|count| *count > 0), "{counts:?}");
    }

    #[test]
    fn kokoro_local_prepare_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");

        assert!(!prepared.phonemes.is_empty());
        assert!(!prepared.token_ids.is_empty());
        assert_eq!(prepared.ref_style.shape().dims(), &[1, 256]);
    }

    #[test]
    fn kokoro_local_bert_prosody_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro prosody smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_bert_prosody_debug(&prepared)
            .expect("run Kokoro BERT/prosody debug");

        assert!(!debug.duration_frames.is_empty());
        assert!(debug.expanded_frames > 0);
        assert_eq!(debug.f0_shape.len(), 2);
        assert_eq!(debug.n_shape.len(), 2);
    }

    #[test]
    fn kokoro_local_predecoder_debug_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro predecoder smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare Kokoro request");
        let debug = model
            .run_predecoder_debug(&prepared)
            .expect("run Kokoro predecoder debug");

        assert_eq!(debug.text_encoder_shape.len(), 3);
        assert_eq!(debug.asr_shape.len(), 3);
        assert!(debug.prosody.expanded_frames > 0);
    }

    #[test]
    fn kokoro_local_generate_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro generate smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let result = model
            .generate("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("run Kokoro generate");

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert!(!result.samples.is_empty());
        assert!(result.samples.iter().all(|v| v.is_finite()));
        assert!(result.samples.len() > 100);
    }

    #[test]
    fn kokoro_local_native_two_row_generate_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };
        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro batch smoke");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let first = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare first Kokoro row");
        let second = model
            .prepare_request("Hello world.", Some("af_heart"), Some("en-US"), 1.0)
            .expect("prepare second Kokoro row");
        let outputs = model
            .generate_prepared_batch(&[first, second])
            .expect("run native Kokoro B=2 generation");
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| !output.samples.is_empty()));
    }

    #[test]
    fn kokoro_local_generate_metal_smoke_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        #[cfg(feature = "metal")]
        let _lease = crate::backends::metal_test_device_or_skip(
            "kokoro_local_generate_metal_smoke_if_env_set",
        );
        let Ok(device) = DeviceSelector::detect_with_preference(Some("metal")) else {
            return;
        };
        if device.kind != DeviceKind::Metal {
            return;
        }

        let model = KokoroTtsModel::load(Path::new(&model_dir), device)
            .expect("load local Kokoro model on Metal");
        let result = model
            .generate(
                "Hello my name is Bella",
                Some("af_bella"),
                Some("en-US"),
                1.0,
            )
            .expect("run Kokoro generate on Metal");

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        assert!(!result.samples.is_empty());
        assert!(result.samples.iter().all(|v| v.is_finite()));
    }

    /// KOK-3 live: punctuation reaches the model as tokens, so pauses and
    /// sentence ends change the (deterministic) durations.
    #[test]
    fn kokoro_local_punctuation_changes_the_audio_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };
        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro punctuation check");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let prepared = model
            .prepare_request("Wait, really? Yes. No!", Some("af_heart"), None, 1.0)
            .expect("prepare punctuated request");
        assert!(prepared.phonemes.ends_with("nˈO!"), "{}", prepared.phonemes);
        let samples = |text: &str| {
            model
                .generate_with_seed_for_test(text, Some("af_heart"), None, 1.0, 7)
                .expect("generate")
                .samples
                .len()
        };
        assert_ne!(
            samples("Wait, really? Yes. No!"),
            samples("Wait really Yes No")
        );
    }

    #[test]
    fn kokoro_local_audio_regression_cpu_if_env_set() {
        let Some(model_dir) = std::env::var_os("IZWI_KOKORO_MODEL_DIR") else {
            return;
        };

        let device = DeviceSelector::detect_with_preference(Some("cpu"))
            .expect("detect cpu device for Kokoro regression");
        let model =
            KokoroTtsModel::load(Path::new(&model_dir), device).expect("load local Kokoro model");
        let result = model
            .generate_with_seed_for_test(
                "Hello my name is Bella",
                Some("af_bella"),
                Some("en-US"),
                1.0,
                0xBEE1_A123_2026u64,
            )
            .expect("run seeded Kokoro regression synthesis");

        let duration_s = result.samples.len() as f32 / result.sample_rate as f32;
        let rms = rms(&result.samples);
        let peak = peak_abs(&result.samples);
        let zcr = zero_crossing_rate(&result.samples);
        let centroid_hz = spectral_centroid_hz(&result.samples, result.sample_rate);

        eprintln!(
            "kokoro regression metrics: len={}, dur={:.3}s, rms={:.6}, peak={:.6}, zcr={:.6}, centroid={:.2}Hz",
            result.samples.len(),
            duration_s,
            rms,
            peak,
            zcr,
            centroid_hz
        );

        assert_eq!(result.sample_rate, KokoroConfig::TARGET_SAMPLE_RATE);
        // misaki-convention phonemes (KOK-3/KOK-4): "həlˈO mI nˈAm ɪz bˈɛlə"
        // has single-token diphthongs, so it is shorter than raw espeak's
        // 52,800 samples.
        assert_eq!(result.samples.len(), 49_800, "unexpected sample length");
        assert!((duration_s - 2.075).abs() < 0.02, "duration_s={duration_s}");
        assert!((rms - 0.047_011).abs() < 0.015, "rms={rms}");
        assert!((peak - 0.373_43).abs() < 0.15, "peak={peak}");
        assert!((zcr - 0.222_845).abs() < 0.08, "zcr={zcr}");
        assert!(
            (centroid_hz - 5_955.96).abs() < 900.0,
            "centroid_hz={centroid_hz}"
        );
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let mean_sq = samples
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            / samples.len() as f64;
        mean_sq.sqrt() as f32
    }

    fn peak_abs(samples: &[f32]) -> f32 {
        samples
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, |a, b| a.max(b))
    }

    fn zero_crossing_rate(samples: &[f32]) -> f32 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            let a = w[0];
            let b = w[1];
            if (a >= 0.0 && b < 0.0) || (a < 0.0 && b >= 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 / (samples.len() - 1) as f32
    }

    fn spectral_centroid_hz(samples: &[f32], sample_rate: u32) -> f32 {
        let n = samples.len().clamp(256, 4096).next_power_of_two().min(4096);
        if n < 2 {
            return 0.0;
        }
        let mut frame = vec![Complex32::new(0.0, 0.0); n];
        for i in 0..n {
            let s = *samples.get(i).unwrap_or(&0.0);
            let w = 0.5f32 - 0.5f32 * ((2.0 * std::f32::consts::PI * i as f32) / n as f32).cos();
            frame[i] = Complex32::new(s * w, 0.0);
        }
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut frame);
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (k, c) in frame.iter().take(n / 2 + 1).enumerate() {
            let mag = c.norm() as f64;
            let hz = k as f64 * sample_rate as f64 / n as f64;
            num += hz * mag;
            den += mag;
        }
        if den <= 1e-12 {
            0.0
        } else {
            (num / den) as f32
        }
    }
}
