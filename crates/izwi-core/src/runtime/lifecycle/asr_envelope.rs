//! The request envelope of the ASR admission framework (ADM-4, PIPE-15): the
//! memory one ASR request needs while it runs, as a function of its audio
//! length and the backend.
//!
//! Every family transcribes long audio in chunks of at most
//! [`AsrEnvelopeGeometry::max_chunk_secs`] (Parakeet in encoder windows), so a
//! request's working set is bounded by one chunk however long the input is:
//! - host: the canonical 16 kHz samples of one chunk with the resampler's and
//!   padder's copies, and the log-mel features of the chunk;
//! - accelerator: the larger of one encoder layer's live set over the chunk
//!   and one decoder layer's prefill live set over the chunk's prompt (the two
//!   run as separate stages), plus one decode logits row.
//!
//! The forced aligner (ADM-5) reuses the formulas for one alignment window:
//! its prompt holds the reference words beside the audio, its classifier
//! scores only the timestamp rows, its portable prefill attends by query
//! blocks, and its convolution stem runs by slices; each window is reserved
//! as a stage workspace when it runs, not held at load.
//!
//! Activations are priced at F32 on every backend (the families compute
//! softmax, norms and logits in F32 even when their weights are half
//! precision), so the envelope is an upper bound for the half-precision
//! paths. Parakeet keeps its own windowed-encoder pricing
//! ([`parakeet_preparation_bytes`]), whose host part grows with the whole
//! utterance because its decode spans all windows. Nemotron adds its sliced
//! convolution stem ([`nemotron_stem_bytes`]) to its encoder layer.
//!
//! The geometry is pinned per published checkpoint; a test checks the
//! dimensions against the checkpoint census fixtures.

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::nemotron::asr::nemotron_stem_bytes;
use crate::models::architectures::parakeet::asr::parakeet_preparation_bytes;
use crate::models::architectures::qwen3::asr::{
    qwen3_aligner_max_words, qwen3_audio_cnn_chunk_bytes, qwen3_audio_cnn_slice_chunks,
    QWEN3_ALIGNER_MAX_AUDIO_SECONDS,
};
use crate::models::architectures::qwen3::core::QWEN3_STATELESS_QUERY_BLOCK;

const F32_BYTES: u64 = 4;
const CANONICAL_SAMPLE_RATE: u64 = 16_000;
/// Log-mel features: 128 bins at a 10 ms hop (the widest front end here).
const MEL_BINS: u64 = 128;
const MEL_FRAMES_PER_SEC: u64 = 100;
/// Copies of one chunk's samples alive while it is prepared: the source
/// slice, the resampled copy and the padded STFT input.
const HOST_SAMPLE_COPIES: u64 = 3;
/// Text template and instruction tokens the planners reserve beside the
/// audio (`ASR_TEXT_PROMPT_TOKEN_RESERVE`).
const PROMPT_TEXT_TOKENS: u64 = 128;
/// `[T, hidden]` activations alive inside a layer at once: the residual, the
/// normed input, Q, K, V and the attention output.
const HIDDEN_PLANES: u64 = 6;
/// `[T, ff]` activations alive inside a feed-forward block (up and gate or
/// activation).
const FF_PLANES: u64 = 2;
/// Attention scores and their softmax.
const SCORE_PLANES: u64 = 2;
/// The portable causal prefill's scores per query block: scaled, masked and
/// softmaxed copies are alive together.
const BLOCKED_SCORE_PLANES: u64 = 3;
/// Fixed host allowance for request bookkeeping, tokenization and buffers
/// the formulas do not name.
const HOST_BASE_BYTES: u64 = 16 * 1024 * 1024;
/// Candle's Metal pool rounds every buffer up to a power of two and reuses a
/// free buffer only when it is at least as large (`buf_size`,
/// `find_available_buffer` in `metal_backend/device.rs`), so a Metal working
/// set can take up to twice its tensor bytes. A 175 s aligner window
/// measured 1.6×; the aligner's and Nemotron's Metal envelopes are priced at
/// the bound.
const METAL_POOL_ROUNDING: u64 = 2;

/// What one ASR request holds while it runs, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct AsrRequestEnvelope {
    pub(crate) host_bytes: u64,
    pub(crate) accelerator_bytes: u64,
}

impl AsrRequestEnvelope {
    /// Both parts in one memory domain (CPU and Metal share one).
    pub(crate) fn total_bytes(self) -> Result<u64> {
        self.host_bytes
            .checked_add(self.accelerator_bytes)
            .ok_or_else(overflow)
    }

    /// `slots` concurrent requests.
    pub(crate) fn scaled(self, slots: usize) -> Result<Self> {
        self.scaled_by(u64::try_from(slots.max(1)).map_err(|_| overflow())?)
    }

    fn scaled_by(self, factor: u64) -> Result<Self> {
        Ok(Self {
            host_bytes: self.host_bytes.checked_mul(factor).ok_or_else(overflow)?,
            accelerator_bytes: self
                .accelerator_bytes
                .checked_mul(factor)
                .ok_or_else(overflow)?,
        })
    }
}

/// One transformer stack's dimensions for the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StackGeometry {
    pub(crate) hidden: u64,
    pub(crate) ff: u64,
    pub(crate) heads: u64,
}

/// The envelope inputs of one ASR checkpoint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AsrEnvelopeGeometry {
    /// The longest audio one encoder/decoder pass takes.
    pub(crate) max_chunk_secs: f32,
    /// Encoder sequence positions per second of audio, at the attention.
    pub(crate) encoder_positions_per_sec: f32,
    /// Encoder attention window when it is local (`None`: the whole chunk).
    pub(crate) encoder_window_secs: Option<f32>,
    pub(crate) encoder: StackGeometry,
    /// Decoder prompt tokens per second of audio (`0` when audio enters
    /// through cross-attention or a transducer).
    pub(crate) audio_tokens_per_sec: f32,
    /// Decoder positions beside the audio tokens: the text prompt for LLM
    /// decoders, the whole target window for Whisper.
    pub(crate) prompt_tokens: u64,
    pub(crate) decoder: StackGeometry,
    pub(crate) vocab: u64,
    /// The audio convolution stem, when it is priced (`None`: folded into
    /// the encoder estimate).
    pub(crate) conv_stem: Option<ConvStemGeometry>,
    /// The forced aligner's text side; `None` for ASR.
    pub(crate) forced_alignment: Option<ForcedAlignmentGeometry>,
}

/// A convolution stem over 1 s feature chunks that runs by slices of whole
/// chunks (`AudioTower::conv_stem`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConvStemGeometry {
    pub(crate) channels: u64,
    pub(crate) mel_bins: u64,
    pub(crate) chunk_frames: u64,
    /// Feature rows (mel bins / 8) and frames per chunk after the stem.
    pub(crate) output_rows: u64,
    pub(crate) output_frames: u64,
    pub(crate) conv_chunksize: usize,
}

/// What a forced-alignment window adds to the ASR formulas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ForcedAlignmentGeometry {
    /// Prompt tokens per aligned word: its text and two timestamp markers.
    pub(crate) tokens_per_word: f32,
    /// Classifier rows per word (its start and end timestamps).
    pub(crate) scored_rows_per_word: u64,
    /// Query rows per block of the portable causal prefill.
    pub(crate) query_block: u64,
}

const fn stack(hidden: u64, ff: u64, heads: u64) -> StackGeometry {
    StackGeometry { hidden, ff, heads }
}

/// The pinned geometry of `variant`, or `None` for a variant that is not an
/// envelope-priced ASR checkpoint (Parakeet is priced by its own windowed
/// formula, see [`request_envelope`]).
pub(crate) fn envelope_geometry(variant: ModelVariant) -> Option<AsrEnvelopeGeometry> {
    // Long-form chunking cuts at `AsrLongFormConfig::default().hard_max_chunk_secs`.
    const CHUNK_SECS: f32 = 30.0;
    Some(match variant {
        // Encoder over 1,500 positions per 30 s window; the decoder sees the
        // audio through cross-attention, with up to 448 target positions.
        ModelVariant::WhisperLargeV3Turbo => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 50.0,
            encoder_window_secs: None,
            encoder: stack(1280, 5120, 20),
            audio_tokens_per_sec: 0.0,
            prompt_tokens: 448,
            decoder: stack(1280, 5120, 20),
            vocab: 51_866,
            conv_stem: None,
            forced_alignment: None,
        },
        // Audio tower after its ×8 convolution stem: 13 positions per second.
        ModelVariant::Qwen3Asr06BGguf => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 13.0,
            encoder_window_secs: None,
            encoder: stack(896, 3584, 14),
            audio_tokens_per_sec: 13.0,
            prompt_tokens: PROMPT_TEXT_TOKENS,
            decoder: stack(1024, 3072, 16),
            vocab: 151_936,
            conv_stem: None,
            forced_alignment: None,
        },
        ModelVariant::Qwen3Asr17BGguf => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 13.0,
            encoder_window_secs: None,
            encoder: stack(1024, 4096, 16),
            audio_tokens_per_sec: 13.0,
            prompt_tokens: PROMPT_TEXT_TOKENS,
            decoder: stack(2048, 6144, 16),
            vocab: 151_936,
            conv_stem: None,
            forced_alignment: None,
        },
        // FastConformer at 12.5 positions per second; an RNN-T prediction
        // network and joint (13,088 outputs) instead of an LLM decoder.
        ModelVariant::Nemotron35AsrStreaming06B => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 12.5,
            encoder_window_secs: None,
            encoder: stack(1024, 4096, 8),
            audio_tokens_per_sec: 0.0,
            prompt_tokens: 0,
            decoder: stack(640, 640, 1),
            vocab: 13_088,
            conv_stem: None,
            forced_alignment: None,
        },
        // Conformer over 50 positions per second in 4 s attention blocks;
        // the projector emits 10 audio tokens per second.
        _ if variant.family() == ModelFamily::GraniteSpeechAsr => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 50.0,
            encoder_window_secs: Some(4.0),
            encoder: stack(1024, 4096, 8),
            audio_tokens_per_sec: 10.0,
            prompt_tokens: PROMPT_TEXT_TOKENS,
            decoder: stack(2048, 8192, 32),
            vocab: 100_353,
            conv_stem: None,
            forced_alignment: None,
        },
        // The retained route takes one 60 s tokenizer chunk (450 audio
        // tokens) and 512 prompt tokens; the σ-VAE tokenizers are
        // convolutional (priced as a 7.5 Hz stack of the decoder width).
        ModelVariant::VibeVoiceAsr => AsrEnvelopeGeometry {
            max_chunk_secs: 60.0,
            encoder_positions_per_sec: 7.5,
            encoder_window_secs: None,
            encoder: stack(3584, 18_944, 28),
            audio_tokens_per_sec: 7.5,
            prompt_tokens: 512,
            decoder: stack(3584, 18_944, 28),
            vocab: 152_064,
            conv_stem: None,
            forced_alignment: None,
        },
        // Causal Whisper-style encoder at 50 positions per second, 12.5
        // audio tokens per second into the decoder.
        ModelVariant::VoxtralMini4BRealtime2602 => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 50.0,
            encoder_window_secs: None,
            encoder: stack(1280, 5120, 32),
            audio_tokens_per_sec: 12.5,
            prompt_tokens: PROMPT_TEXT_TOKENS,
            decoder: stack(3072, 9216, 32),
            vocab: 131_072,
            conv_stem: None,
            forced_alignment: None,
        },
        // FastConformer at 12.5 positions per second into the LFM2 backbone.
        ModelVariant::Lfm25Audio15BGguf => AsrEnvelopeGeometry {
            max_chunk_secs: CHUNK_SECS,
            encoder_positions_per_sec: 12.5,
            encoder_window_secs: None,
            encoder: stack(512, 2048, 8),
            audio_tokens_per_sec: 12.5,
            prompt_tokens: PROMPT_TEXT_TOKENS,
            decoder: stack(2048, 8192, 32),
            vocab: 65_536,
            conv_stem: None,
            forced_alignment: None,
        },
        // The audio tower of Qwen3-ASR-1.7B (104-position ≈ 8 s attention
        // windows after the ×8 stem) under a 5,000-class timestamp head; the
        // prompt is the window's audio and about 3.5 tokens per word.
        _ if variant.family() == ModelFamily::Qwen3ForcedAligner => AsrEnvelopeGeometry {
            max_chunk_secs: QWEN3_ALIGNER_MAX_AUDIO_SECONDS as f32,
            encoder_positions_per_sec: 13.0,
            encoder_window_secs: Some(8.0),
            encoder: stack(1024, 4096, 16),
            audio_tokens_per_sec: 13.0,
            // `<|audio_start|>` and `<|audio_end|>`.
            prompt_tokens: 2,
            decoder: stack(1024, 3072, 16),
            vocab: 5_000,
            conv_stem: Some(ConvStemGeometry {
                channels: 480,
                mel_bins: MEL_BINS,
                chunk_frames: MEL_FRAMES_PER_SEC,
                output_rows: MEL_BINS / 8,
                output_frames: 13,
                conv_chunksize: 500,
            }),
            forced_alignment: Some(ForcedAlignmentGeometry {
                tokens_per_word: 3.5,
                scored_rows_per_word: 2,
                query_block: QWEN3_STATELESS_QUERY_BLOCK as u64,
            }),
        },
        _ => return None,
    })
}

/// Parakeet's encoder window: 1,024 frames (81.92 s) plus 128 frames of
/// context on each side (`EncoderWindowing::V3`).
const PARAKEET_WINDOW_SECS: f32 = 102.4;

/// The longest audio one pass of `variant` takes, or `None` for a variant
/// without an envelope.
pub(crate) fn max_chunk_secs(variant: ModelVariant) -> Option<f32> {
    if variant == ModelVariant::ParakeetTdt06BV3 {
        return Some(PARAKEET_WINDOW_SECS);
    }
    envelope_geometry(variant).map(|geometry| geometry.max_chunk_secs)
}

/// Planners cut long audio at 95% of the solved invocation limit
/// (`NativeExecutor::asr_chunk_plan`), so a full 30 s chunk needs a context
/// sized for `30 / 0.95` seconds.
const PLANNED_CHUNK_SECS: f32 = 30.0 / 0.95;

/// The smallest decoder context, in tokens, one `variant` sequence needs for
/// one full chunk: its text prompt, the chunk's audio tokens and its decode
/// budget (P2.5, ADM-10). A state fit must reach at least this, or refuse
/// the load: a smaller published context cannot hold a default request
/// (an LFM2.5 Audio fit published 320 tokens, below a 30 s prompt of 391).
/// `None` for families without an LLM decoder context (Parakeet, Nemotron).
pub(crate) fn minimum_sequence_tokens(variant: ModelVariant) -> Option<u64> {
    use crate::runtime::{asr_sequence_ceiling, ASR_TEXT_PROMPT_TOKEN_RESERVE as PROMPT};
    let geometry = envelope_geometry(variant)?;
    let rate = geometry.audio_tokens_per_sec;
    let tokens = match variant {
        // The decoder's whole target window (prompt and output).
        ModelVariant::WhisperLargeV3Turbo => geometry.prompt_tokens as usize,
        // Single-pass planning reserves the default generation budget.
        ModelVariant::Qwen3Asr06BGguf | ModelVariant::Qwen3Asr17BGguf => {
            asr_sequence_ceiling(PROMPT, rate, PLANNED_CHUNK_SECS, 768)
        }
        _ if variant.family() == ModelFamily::GraniteSpeechAsr => asr_sequence_ceiling(
            PROMPT,
            rate,
            PLANNED_CHUNK_SECS,
            crate::runtime::granite_auto_asr_max_tokens_for_duration(30.0),
        ),
        // The retained route's one 60 s tokenizer chunk.
        ModelVariant::VibeVoiceAsr => asr_sequence_ceiling(
            geometry.prompt_tokens as usize,
            rate,
            geometry.max_chunk_secs,
            crate::models::architectures::vibevoice::asr::vibevoice_auto_asr_max_tokens_for_duration(
                geometry.max_chunk_secs,
            ),
        ),
        // The realtime decoder emits about one token per audio token.
        ModelVariant::VoxtralMini4BRealtime2602 => asr_sequence_ceiling(
            PROMPT,
            rate,
            PLANNED_CHUNK_SECS,
            (rate * PLANNED_CHUNK_SECS).ceil() as usize,
        ),
        // LFM2.5 Audio cuts 30 s chunks itself and decodes up to its
        // single-pass default.
        ModelVariant::Lfm25Audio15BGguf => asr_sequence_ceiling(
            PROMPT,
            rate,
            geometry.max_chunk_secs,
            crate::models::registry::LFM25_AUDIO_ASR_DEFAULT_MAX_NEW_TOKENS,
        ),
        _ => return None,
    };
    u64::try_from(tokens).ok()
}

fn overflow() -> Error {
    Error::Overloaded("ASR request envelope overflow".into())
}

fn product(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(1u64, |total, value| total.checked_mul(*value))
        .ok_or_else(overflow)
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(overflow)
}

fn positions(rate: f32, secs: f32) -> u64 {
    (rate * secs).ceil().max(0.0) as u64
}

/// One layer's live activations over `length` positions attending over
/// `context` positions.
fn layer_bytes(geometry: StackGeometry, length: u64, context: u64) -> Result<u64> {
    sum(&[
        product(&[HIDDEN_PLANES, length, geometry.hidden, F32_BYTES])?,
        product(&[FF_PLANES, length, geometry.ff, F32_BYTES])?,
        product(&[SCORE_PLANES, geometry.heads, length, context, F32_BYTES])?,
    ])
}

/// One causal prefill layer over `length` positions whose attention runs
/// `block` query rows at a time against the keys before them.
fn blocked_prefill_bytes(geometry: StackGeometry, length: u64, block: u64) -> Result<u64> {
    sum(&[
        product(&[HIDDEN_PLANES, length, geometry.hidden, F32_BYTES])?,
        product(&[FF_PLANES, length, geometry.ff, F32_BYTES])?,
        product(&[
            BLOCKED_SCORE_PLANES,
            geometry.heads,
            block.min(length),
            length,
            F32_BYTES,
        ])?,
    ])
}

impl ConvStemGeometry {
    /// The stem's peak over `chunks` chunks: one slice's transient beside
    /// the slice outputs already produced, and those outputs with their
    /// concatenation.
    fn bytes(self, chunks: u64) -> Result<u64> {
        let slice = qwen3_audio_cnn_slice_chunks(
            self.channels,
            self.mel_bins,
            self.chunk_frames,
            Some(self.conv_chunksize),
        ) as u64;
        let transient = product(&[
            slice.min(chunks),
            qwen3_audio_cnn_chunk_bytes(self.channels, self.mel_bins, self.chunk_frames),
        ])?;
        let outputs = product(&[
            2,
            chunks,
            self.channels,
            self.output_rows,
            self.output_frames,
            F32_BYTES,
        ])?;
        sum(&[transient, outputs])
    }
}

/// The envelope of one `variant` request over `audio_secs` of audio on
/// `backend`, or `None` for a variant without one. Audio beyond one chunk
/// adds nothing (it is chunked), except Parakeet's host features and
/// stitched encoder output, which span the whole utterance and grow linearly.
pub(crate) fn request_envelope(
    variant: ModelVariant,
    audio_secs: f32,
    backend: BackendKind,
) -> Result<Option<AsrRequestEnvelope>> {
    let audio_secs = if audio_secs.is_finite() {
        audio_secs.max(0.0)
    } else {
        0.0
    };
    if variant == ModelVariant::ParakeetTdt06BV3 {
        let samples = usize::try_from(positions(CANONICAL_SAMPLE_RATE as f32, audio_secs))
            .map_err(|_| overflow())?;
        let bytes = parakeet_preparation_bytes(samples, backend)?;
        return Ok(Some(AsrRequestEnvelope {
            host_bytes: sum(&[HOST_BASE_BYTES, bytes.host_bytes])?,
            accelerator_bytes: bytes.device_bytes()?,
        }));
    }
    let Some(geometry) = envelope_geometry(variant) else {
        return Ok(None);
    };
    // Without the transcript, a window holds the most words it accepts.
    let words = geometry
        .forced_alignment
        .map(|_| qwen3_aligner_max_words(f64::from(audio_secs.min(geometry.max_chunk_secs))));
    let mut envelope = geometry_envelope(geometry, audio_secs, words, backend)?;
    if variant == ModelVariant::Nemotron35AsrStreaming06B {
        envelope.accelerator_bytes = nemotron_accelerator_bytes(
            envelope.accelerator_bytes,
            audio_secs.min(geometry.max_chunk_secs),
            backend,
        )?;
    }
    Ok(Some(envelope))
}

/// Nemotron's stem runs before its encoder layers, in time slices
/// ([`nemotron_stem_bytes`]), and Candle's Metal pool keeps the stem's
/// buffers while the layers run, so the stem adds to the encoder layer's
/// live set. On Metal both are priced at the pool-rounding bound: a 30 s
/// pass grew device memory by 211 MB, against 256 MB priced.
fn nemotron_accelerator_bytes(encoder_bytes: u64, secs: f32, backend: BackendKind) -> Result<u64> {
    let frames =
        usize::try_from(positions(MEL_FRAMES_PER_SEC as f32, secs)).map_err(|_| overflow())?;
    let bytes = sum(&[encoder_bytes, nemotron_stem_bytes(frames)?])?;
    if backend == BackendKind::Metal {
        product(&[bytes, METAL_POOL_ROUNDING])
    } else {
        Ok(bytes)
    }
}

/// The working set of one forced-alignment window of `audio_secs` holding
/// `words` words on `backend`, or `None` for a variant that is not a forced
/// aligner.
pub(crate) fn forced_alignment_envelope(
    variant: ModelVariant,
    audio_secs: f32,
    words: usize,
    backend: BackendKind,
) -> Result<Option<AsrRequestEnvelope>> {
    let Some(geometry) = envelope_geometry(variant).filter(|g| g.forced_alignment.is_some()) else {
        return Ok(None);
    };
    let audio_secs = if audio_secs.is_finite() {
        audio_secs.max(0.0)
    } else {
        0.0
    };
    geometry_envelope(geometry, audio_secs, Some(words), backend).map(Some)
}

fn geometry_envelope(
    geometry: AsrEnvelopeGeometry,
    audio_secs: f32,
    words: Option<usize>,
    backend: BackendKind,
) -> Result<AsrRequestEnvelope> {
    let secs = audio_secs.min(geometry.max_chunk_secs);
    let samples = positions(CANONICAL_SAMPLE_RATE as f32, secs);
    let host_bytes = sum(&[
        HOST_BASE_BYTES,
        product(&[HOST_SAMPLE_COPIES, samples, F32_BYTES])?,
        product(&[
            2,
            MEL_BINS,
            positions(MEL_FRAMES_PER_SEC as f32, secs),
            F32_BYTES,
        ])?,
    ])?;
    let encoder_length = positions(geometry.encoder_positions_per_sec, secs).max(1);
    let encoder_context = geometry
        .encoder_window_secs
        .map_or(encoder_length, |window| {
            positions(geometry.encoder_positions_per_sec, window).min(encoder_length)
        })
        .max(1);
    let encoder = layer_bytes(geometry.encoder, encoder_length, encoder_context)?;
    let words = u64::try_from(words.unwrap_or(0)).map_err(|_| overflow())?;
    let text_tokens = geometry.forced_alignment.map_or(0, |alignment| {
        (alignment.tokens_per_word * words as f32).ceil() as u64
    });
    let prompt = sum(&[
        positions(geometry.audio_tokens_per_sec, secs),
        geometry.prompt_tokens,
        text_tokens,
    ])?
    .max(1);
    let prefill = match geometry.forced_alignment {
        Some(alignment) => blocked_prefill_bytes(geometry.decoder, prompt, alignment.query_block)?,
        None => layer_bytes(geometry.decoder, prompt, prompt)?,
    };
    let stem = match geometry.conv_stem {
        Some(stem) => stem.bytes(secs.ceil().max(1.0) as u64)?,
        None => 0,
    };
    // ASR reads one decode row; the aligner its timestamp rows.
    let logits_rows = geometry.forced_alignment.map_or(Ok(1), |alignment| {
        product(&[alignment.scored_rows_per_word, words]).map(|rows| rows.max(1))
    })?;
    let logits = product(&[logits_rows, geometry.vocab, F32_BYTES])?;
    let mut accelerator_bytes = sum(&[encoder.max(prefill).max(stem), logits])?;
    // The ASR families keep their established Metal pricing.
    if geometry.forced_alignment.is_some() && backend == BackendKind::Metal {
        accelerator_bytes = product(&[accelerator_bytes, METAL_POOL_ROUNDING])?;
    }
    Ok(AsrRequestEnvelope {
        host_bytes,
        accelerator_bytes,
    })
}

/// What load holds back from the state fits for `request_slots` concurrent
/// requests, each on one full chunk (Parakeet: one full encoder window).
/// `None` for the forced aligner: it has no state fit, and each alignment
/// window reserves its own envelope as a stage workspace.
pub(crate) fn deferred_request_envelope(
    variant: ModelVariant,
    backend: BackendKind,
    request_slots: usize,
) -> Result<Option<AsrRequestEnvelope>> {
    if envelope_geometry(variant).is_some_and(|geometry| geometry.forced_alignment.is_some()) {
        return Ok(None);
    }
    let Some(secs) = max_chunk_secs(variant) else {
        return Ok(None);
    };
    let Some(envelope) = request_envelope(variant, secs, backend)? else {
        return Ok(None);
    };
    let slots = u64::try_from(request_slots.max(1)).map_err(|_| overflow())?;
    envelope.scaled_by(slots).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENVELOPE_VARIANTS: [ModelVariant; 10] = [
        ModelVariant::WhisperLargeV3Turbo,
        ModelVariant::Qwen3Asr06BGguf,
        ModelVariant::Qwen3Asr17BGguf,
        ModelVariant::ParakeetTdt06BV3,
        ModelVariant::Nemotron35AsrStreaming06B,
        ModelVariant::GraniteSpeech412BPlus,
        ModelVariant::GraniteSpeech412B,
        ModelVariant::VibeVoiceAsr,
        ModelVariant::VoxtralMini4BRealtime2602,
        ModelVariant::Lfm25Audio15BGguf,
    ];

    fn envelope(variant: ModelVariant, secs: f32) -> AsrRequestEnvelope {
        request_envelope(variant, secs, BackendKind::Metal)
            .unwrap()
            .expect("ASR envelope")
    }

    /// Envelopes grow with audio length, stop growing at one chunk (Parakeet:
    /// only its whole-utterance host features and stitched output keep
    /// growing, linearly), and
    /// stay within a request a 16 GB machine can run.
    #[test]
    fn envelopes_are_monotonic_and_bounded_by_one_chunk() {
        const GIB: u64 = 1024 * 1024 * 1024;
        for variant in ENVELOPE_VARIANTS {
            let chunk = max_chunk_secs(variant).expect("chunk");
            let mut previous = AsrRequestEnvelope::default();
            for secs in [0.5, 5.0, 15.0, chunk * 0.5, chunk] {
                let current = envelope(variant, secs);
                assert!(
                    current.host_bytes >= previous.host_bytes
                        && current.accelerator_bytes >= previous.accelerator_bytes,
                    "{variant}: {secs} s shrank: {previous:?} -> {current:?}"
                );
                previous = current;
            }
            let at_chunk = envelope(variant, chunk);
            let at_hour = envelope(variant, 3_600.0);
            if variant != ModelVariant::ParakeetTdt06BV3 {
                assert_eq!(at_hour, at_chunk, "{variant}");
            }
            assert!(
                at_chunk.total_bytes().unwrap() < 2 * GIB,
                "{variant}: {at_chunk:?}"
            );
            assert!(at_chunk.accelerator_bytes > 0 && at_chunk.host_bytes > 0);
            assert_eq!(
                deferred_request_envelope(variant, BackendKind::Metal, 2)
                    .unwrap()
                    .unwrap(),
                at_chunk.scaled(2).unwrap()
            );
        }
        assert_eq!(
            request_envelope(ModelVariant::Kokoro82M, 10.0, BackendKind::Cpu).unwrap(),
            None
        );
        assert_eq!(
            envelope(ModelVariant::WhisperLargeV3Turbo, f32::NAN),
            envelope(ModelVariant::WhisperLargeV3Turbo, 0.0)
        );

        // The forced aligner (ADM-5): one window, at its most words, grows
        // with the window and stops at the 180 s limit, within 2 GiB on Metal
        // (pool rounding included). It is reserved per window stage, never
        // held at load.
        for variant in ALIGNER_VARIANTS {
            let window = max_chunk_secs(variant).expect("aligner window");
            assert_eq!(window, 180.0);
            let mut previous = AsrRequestEnvelope::default();
            for secs in [0.5, 5.0, 15.0, 30.0, 90.0, window] {
                let current = envelope(variant, secs);
                assert!(
                    current.host_bytes > previous.host_bytes
                        && current.accelerator_bytes > previous.accelerator_bytes,
                    "{variant}: {secs} s did not grow: {previous:?} -> {current:?}"
                );
                previous = current;
            }
            let at_window = envelope(variant, window);
            assert_eq!(envelope(variant, 3_600.0), at_window, "{variant}");
            assert!(
                at_window.total_bytes().unwrap() < 2 * GIB,
                "{variant}: {at_window:?}"
            );
            assert_eq!(
                deferred_request_envelope(variant, BackendKind::Metal, 2).unwrap(),
                None
            );
        }
    }

    const ALIGNER_VARIANTS: [ModelVariant; 2] = [
        ModelVariant::Qwen3ForcedAligner06B,
        ModelVariant::Qwen3ForcedAligner06B4Bit,
    ];

    /// A window's envelope follows its words and seconds; the 180 s window
    /// at its word bound prices the slice-bounded convolution stem, the
    /// query-blocked prefill and only the timestamp logits rows.
    #[test]
    fn forced_alignment_envelope_follows_words_and_seconds() {
        const MIB: u64 = 1024 * 1024;
        let variant = ModelVariant::Qwen3ForcedAligner06B;
        let window = |secs: f32, words: usize| {
            forced_alignment_envelope(variant, secs, words, BackendKind::Cpu)
                .unwrap()
                .expect("aligner envelope")
        };
        assert!(window(30.0, 80).accelerator_bytes > window(30.0, 20).accelerator_bytes);
        assert!(window(60.0, 80).accelerator_bytes > window(30.0, 80).accelerator_bytes);
        assert!(window(60.0, 80).host_bytes > window(30.0, 80).host_bytes);
        assert_eq!(window(f32::NAN, 10), window(0.0, 10));
        assert_eq!(
            forced_alignment_envelope(ModelVariant::Qwen3Asr06BGguf, 30.0, 80, BackendKind::Cpu)
                .unwrap(),
            None
        );
        for secs in [30.0f32, 60.0, 120.0, 180.0] {
            let words = qwen3_aligner_max_words(f64::from(secs));
            let bound = window(secs, words);
            let typical = window(secs, (secs * 2.7).ceil() as usize);
            let metal = forced_alignment_envelope(variant, secs, words, BackendKind::Metal)
                .unwrap()
                .unwrap();
            assert_eq!(metal.accelerator_bytes, 2 * bound.accelerator_bytes);
            assert_eq!(metal.host_bytes, bound.host_bytes);
            eprintln!(
                "aligner envelope {secs} s: CPU {} MiB at {words} words, {} MiB at 2.7 words/s; \
                 Metal {} MiB at {words} words",
                bound.total_bytes().unwrap() / MIB,
                typical.total_bytes().unwrap() / MIB,
                metal.total_bytes().unwrap() / MIB
            );
        }
        // 180 s at the 10 words/s bound: an 8,642-token prompt.
        let bound = window(180.0, 1_800);
        let geometry = envelope_geometry(variant).unwrap();
        let prefill = blocked_prefill_bytes(geometry.decoder, 8_642, 256).unwrap();
        let stem = geometry.conv_stem.unwrap().bytes(180).unwrap();
        let logits = 2 * 1_800 * 5_000 * 4;
        assert!(stem > 300 * MIB, "{stem}");
        assert_eq!(bound.accelerator_bytes, prefill.max(stem) + logits);
    }

    /// Nemotron prices one stem slice beside its encoder layer, doubled on
    /// Metal for pool rounding: a 30 s pass measured 201 MiB of device
    /// growth (420 MiB of it was the stem before it was sliced).
    #[test]
    fn nemotron_envelope_prices_one_stem_slice_beside_the_encoder() {
        const MIB: u64 = 1024 * 1024;
        let variant = ModelVariant::Nemotron35AsrStreaming06B;
        let cpu = request_envelope(variant, 30.0, BackendKind::Cpu)
            .unwrap()
            .unwrap();
        let metal = envelope(variant, 30.0);
        let geometry = envelope_geometry(variant).unwrap();
        let encoder = layer_bytes(geometry.encoder, 375, 375).unwrap();
        // One slice of 120 frames and its 2-cell carry: 976 feature frames.
        let stem = 3 * 256 * 489 * 65 * 4;
        assert_eq!(nemotron_stem_bytes(3_000).unwrap(), stem);
        assert_eq!(nemotron_stem_bytes(10_000).unwrap(), stem);
        assert_eq!(cpu.accelerator_bytes, encoder + 13_088 * 4 + stem);
        assert_eq!(metal.accelerator_bytes, 2 * cpu.accelerator_bytes);
        assert!(metal.accelerator_bytes > 201 * MIB, "{metal:?}");
        assert_eq!(metal.host_bytes, cpu.host_bytes);
    }

    /// One chunk per sequence: a default request's prompt and decode budget.
    #[test]
    fn minimum_sequence_tokens_hold_one_chunk() {
        let floor = |variant| minimum_sequence_tokens(variant).unwrap();
        assert_eq!(floor(ModelVariant::Qwen3Asr06BGguf), 1_307);
        assert_eq!(floor(ModelVariant::VibeVoiceAsr), 1_938);
        assert_eq!(floor(ModelVariant::WhisperLargeV3Turbo), 448);
        // A 30 s LFM prompt is about 391 tokens; the floor adds its 1,024
        // single-pass decode budget.
        assert_eq!(floor(ModelVariant::Lfm25Audio15BGguf), 128 + 375 + 1_024);
        assert!(floor(ModelVariant::GraniteSpeech412BPlus) > 128 + 316);
        assert!(floor(ModelVariant::VoxtralMini4BRealtime2602) > 2 * 395);
        assert_eq!(
            minimum_sequence_tokens(ModelVariant::ParakeetTdt06BV3),
            None
        );
        assert_eq!(
            minimum_sequence_tokens(ModelVariant::Nemotron35AsrStreaming06B),
            None
        );
    }

    /// The pinned dimensions match the published checkpoints' tensor shapes.
    #[test]
    fn pinned_geometry_matches_the_checkpoint_census() {
        fn shape(fixture: &str, name: &str) -> Vec<u64> {
            let value: serde_json::Value = serde_json::from_str(fixture).unwrap();
            value["files"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|file| file["tensors"].as_array().unwrap())
                .find(|tensor| tensor[0] == name)
                .unwrap_or_else(|| panic!("{name} missing"))[2]
                .as_array()
                .unwrap()
                .iter()
                .map(|dim| dim.as_u64().unwrap())
                .collect()
        }
        macro_rules! fixture {
            ($file:literal) => {
                include_str!(concat!("../../../tests/fixtures/asr_residency/", $file))
            };
        }
        let whisper = envelope_geometry(ModelVariant::WhisperLargeV3Turbo).unwrap();
        let fixture = fixture!("whisper_large_v3_turbo.json");
        assert_eq!(
            shape(fixture, "model.decoder.embed_tokens.weight"),
            [whisper.vocab, whisper.decoder.hidden]
        );
        assert_eq!(
            shape(fixture, "model.encoder.embed_positions.weight"),
            [
                (whisper.encoder_positions_per_sec * whisper.max_chunk_secs) as u64,
                whisper.encoder.hidden
            ]
        );
        for (variant, fixture) in [
            (
                ModelVariant::Qwen3Asr06BGguf,
                fixture!("qwen3_asr_0_6b_q8_0.json"),
            ),
            (
                ModelVariant::Qwen3Asr17BGguf,
                fixture!("qwen3_asr_1_7b_q8_0.json"),
            ),
        ] {
            let geometry = envelope_geometry(variant).unwrap();
            assert_eq!(
                shape(fixture, "thinker.model.embed_tokens.weight"),
                [geometry.vocab, geometry.decoder.hidden]
            );
            assert_eq!(
                shape(fixture, "thinker.model.layers.{i}.mlp.gate_proj.weight"),
                [geometry.decoder.ff, geometry.decoder.hidden]
            );
            assert_eq!(
                shape(fixture, "thinker.audio_tower.layers.{i}.fc1.weight"),
                [geometry.encoder.ff, geometry.encoder.hidden]
            );
        }
        let nemotron = envelope_geometry(ModelVariant::Nemotron35AsrStreaming06B).unwrap();
        assert_eq!(
            shape(
                fixture!("nemotron_3_5_asr_streaming_0_6b.json"),
                "joint.joint_net.2.weight"
            ),
            [nemotron.vocab, nemotron.decoder.hidden]
        );
        let granite = envelope_geometry(ModelVariant::GraniteSpeech412BPlus).unwrap();
        assert_eq!(
            shape(
                fixture!("granite_speech_4_1_2b_plus.json"),
                "language_model.model.embed_tokens.weight"
            ),
            [granite.vocab, granite.decoder.hidden]
        );
        let vibevoice = envelope_geometry(ModelVariant::VibeVoiceAsr).unwrap();
        assert_eq!(
            shape(fixture!("vibevoice_asr.json"), "lm_head.weight"),
            [vibevoice.vocab, vibevoice.decoder.hidden]
        );
        let voxtral = envelope_geometry(ModelVariant::VoxtralMini4BRealtime2602).unwrap();
        assert_eq!(
            shape(
                fixture!("voxtral_mini_4b_realtime_2602.json"),
                "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
            ),
            [voxtral.vocab, voxtral.decoder.hidden]
        );
        let lfm = envelope_geometry(ModelVariant::Lfm25Audio15BGguf).unwrap();
        assert_eq!(
            shape(fixture!("lfm2_5_audio_1_5b_q4_0.json"), "token_embd.weight"),
            [lfm.vocab, lfm.decoder.hidden]
        );
        let aligner = envelope_geometry(ModelVariant::Qwen3ForcedAligner06B).unwrap();
        let fixture = fixture!("qwen3_forced_aligner_0_6b.json");
        assert_eq!(
            shape(fixture, "thinker.lm_head.weight"),
            [aligner.vocab, aligner.decoder.hidden]
        );
        assert_eq!(
            shape(fixture, "thinker.model.layers.{i}.mlp.gate_proj.weight"),
            [aligner.decoder.ff, aligner.decoder.hidden]
        );
        assert_eq!(
            shape(fixture, "thinker.audio_tower.layers.{i}.fc1.weight"),
            [aligner.encoder.ff, aligner.encoder.hidden]
        );
        let stem = aligner.conv_stem.unwrap();
        assert_eq!(
            shape(fixture, "thinker.audio_tower.conv_out.weight"),
            [aligner.encoder.hidden, stem.channels * stem.output_rows]
        );
    }
}
