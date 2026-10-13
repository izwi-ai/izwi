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
//! Activations are priced at F32 on every backend (the families compute
//! softmax, norms and logits in F32 even when their weights are half
//! precision), so the envelope is an upper bound for the half-precision
//! paths. Parakeet keeps its own windowed-encoder pricing
//! ([`parakeet_preparation_bytes`]), whose host part grows with the whole
//! utterance because its decode spans all windows.
//!
//! The geometry is pinned per published checkpoint; a test checks the
//! dimensions against the checkpoint census fixtures.

use crate::backends::BackendKind;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::parakeet::asr::parakeet_preparation_bytes;

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
/// Fixed host allowance for request bookkeeping, tokenization and buffers
/// the formulas do not name.
const HOST_BASE_BYTES: u64 = 16 * 1024 * 1024;

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

    fn scaled(self, factor: u64) -> Result<Self> {
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
    let prompt = positions(geometry.audio_tokens_per_sec, secs)
        .checked_add(geometry.prompt_tokens)
        .ok_or_else(overflow)?
        .max(1);
    let prefill = layer_bytes(geometry.decoder, prompt, prompt)?;
    let logits_row = product(&[geometry.vocab, F32_BYTES])?;
    Ok(Some(AsrRequestEnvelope {
        host_bytes,
        accelerator_bytes: sum(&[encoder.max(prefill), logits_row])?,
    }))
}

/// What load holds back from the state fits for `request_slots` concurrent
/// requests, each on one full chunk (Parakeet: one full encoder window).
pub(crate) fn deferred_request_envelope(
    variant: ModelVariant,
    backend: BackendKind,
    request_slots: usize,
) -> Result<Option<AsrRequestEnvelope>> {
    let Some(secs) = max_chunk_secs(variant) else {
        return Ok(None);
    };
    let Some(envelope) = request_envelope(variant, secs, backend)? else {
        return Ok(None);
    };
    let slots = u64::try_from(request_slots.max(1)).map_err(|_| overflow())?;
    envelope.scaled(slots).map(Some)
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
    }
}
