//! ASR runtime methods routed through the unified core engine.

use std::collections::HashSet;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use crate::backends::state::{PhysicalStateSequenceId, PhysicalStateTransactionId};
use crate::backends::BackendKind;
use crate::catalog::{parse_model_variant, resolve_asr_model_variant, ModelFamily};
use crate::config::RealtimeAsrSessionConfig;
use crate::engine::{
    AsrProgress, AsrProgressPhase, AsrTaskOptions, EngineCoreRequest, EngineOutput,
    RealtimeAsrSessionHandle, ResourceAmount, ResourceVector, RetainedTensorStateRuntimeV2,
    StreamingOutput, TaskType, WorkCost, WorkUnit,
};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::models::architectures::granite_speech::asr::{
    granite_speech_request_plan, parse_granite_speech_output,
    validate_granite_speech_request_language, GraniteSpeechTask,
};
use crate::models::architectures::granite_speech5::asr::{
    granite_speech5_auto_max_tokens_for_duration, resolve_granite_speech5_source_language,
    validate_granite_speech5_request, GraniteSpeech5RequestOptions,
    GRANITE_SPEECH5_AUTO_MAX_TOKENS,
};
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimePreparationBatchGeometry, VoxtralRealtimePreparationGeometry,
    VoxtralRealtimePreparationMode, VoxtralRealtimePreparationStageSeal,
    VoxtralRealtimePreparedResourceUsage,
};
use crate::models::registry::{
    NativeAsrGenerationOptions, NativeAsrModel, NativeAsrRealtimeEvent,
    NativeAsrRealtimeResourceReservation, NativeAsrRealtimeState, NativeAsrTranscription,
    VoxtralModelLease,
};
use crate::runtime::adapters::{
    capability_row, CapabilityKind, ExecutionTargetKind, LoadedCapabilityBinding,
    LoadedExecutionContract, RealtimeAsrPath, RuntimeAdapterRegistry, StreamingMode,
};
use crate::runtime::audio_io::{
    base64_decode, decode_audio_bytes_canonical, validate_base64_audio_retained_size,
    validate_base64_audio_source_size, MAX_AUDIO_SOURCE_BYTES,
};
use crate::runtime::coordinator::{InferenceCoordinator, JobLease, JobResourceObservation};
use crate::runtime::diarization::samples_to_ms;
use crate::runtime::forced_alignment::{
    align_under_stage_lease, align_window_checked, forced_alignment_window_workspace,
    sample_windows, AlignedWindow, AlignmentWindow, LongTextPlanner, SampleWindow,
    WindowedAlignment,
};
use crate::runtime::request::AsrRuntimeRequest;
#[cfg(test)]
use crate::runtime::service::retained_engine_request_input_bytes;
use crate::runtime::service::{
    copy_optional_preparation_string, copy_preparation_bytes, copy_preparation_string,
    CapabilityLoadOptions, AdmittedEngineRequest, RuntimeService,
};
use crate::runtime::types::{
    AsrTranscription, RuntimeRequestContext, SpeakerAttributedAsrResult,
    SpeakerAttributedAsrStatus, SpeakerAttributedAsrTurn,
};
use crate::runtime::CoordinatorLane;
use crate::serve_runtime::{
    ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS, ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
    ENV_ASR_REALTIME_MAX_SESSIONS,
};
use izwi_asr_toolkit::{clean_transcript_text, plan_audio_chunks, AsrLongFormConfig, AudioChunk};

mod realtime;
mod realtime_push;
mod realtime_session;
mod saa_seam;
mod speaker_attribution;
mod transcribe;
use realtime_push::{RealtimePush, RealtimePushBatcher, RealtimePushHolds, RealtimePushRow};
use realtime_session::{
    add_realtime_stream_reservation, map_native_realtime_events, realtime_asr_execution_path,
    realtime_segment_rollover_due, realtime_state_observation,
    realtime_state_with_input_observation, realtime_stream_resource_vector,
    realtime_trailing_quiet_samples, release_realtime_asr_cleanup, run_realtime_blocking_operation,
    spawn_realtime_asr_watchdog, validate_realtime_input_copy,
    voxtral_realtime_committed_observation, voxtral_realtime_preparation_cost,
    RealtimeAsrEngineOperationHandles, RealtimeAsrOperationGuard, RealtimeAsrOperationHandles,
    RealtimeAsrSessionLease, RealtimeEngineSegment, RuntimeAsrRealtimeResources,
    RuntimeAsrRealtimeState, RuntimeEngineRealtimeSession,
};
#[cfg(test)]
use realtime_session::{
    realtime_asr_session_expiration, voxtral_realtime_preparation_cost_from_geometry,
    RealtimeAsrCleanupGuard, RealtimeAsrDetachedResources, REALTIME_SEGMENT_PAUSE_MS,
    REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR,
};
#[cfg(test)]
use speaker_attribution::{
    granite_saa_chunk_plan, granite_saa_chunk_progress, granite_saa_long_form_config,
    granite_saa_memory_limit_warning,
    granite_saa_decode_diagnostics, granite_saa_max_new_tokens_for_duration,
    granite_saa_max_token_warning, granite_saa_processing_progress,
    resolve_speaker_attributed_asr_variant, speaker_attributed_asr_result_from_text,
    speaker_attributed_asr_result_from_text_with_warnings, GraniteSaaChunkDecodeDiagnostics,
    GraniteSaaTranscriptAssembler,
};

#[derive(Clone, Copy)]
enum AsrAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

enum OwnedAsrAudioInput {
    Base64(String),
    Bytes(Vec<u8>),
}

impl AsrAudioInput<'_> {
    fn input_bytes(self) -> usize {
        match self {
            Self::Base64(audio) => audio.len(),
            Self::Bytes(audio) => audio.len(),
        }
    }

    fn validate_retained_size(self) -> Result<()> {
        match self {
            Self::Base64(audio) => {
                validate_base64_audio_retained_size(audio.len(), MAX_AUDIO_SOURCE_BYTES)
            }
            Self::Bytes([]) => Err(Error::InvalidInput(
                "ASR request missing audio bytes".to_string(),
            )),
            Self::Bytes(audio) if audio.len() > MAX_AUDIO_SOURCE_BYTES => {
                Err(Error::InvalidInput(format!(
                    "ASR encoded audio is {} bytes, exceeding the {MAX_AUDIO_SOURCE_BYTES}-byte source limit",
                    audio.len()
                )))
            }
            Self::Bytes(_) => Ok(()),
        }
    }

    async fn into_owned_for_job(self, job: &JobLease) -> Result<OwnedAsrAudioInput> {
        match self {
            Self::Base64(audio) => Ok(OwnedAsrAudioInput::Base64(
                copy_preparation_string(job, audio, "ASR base64 audio").await?,
            )),
            Self::Bytes(audio) => Ok(OwnedAsrAudioInput::Bytes(
                copy_preparation_bytes(job, audio, "ASR encoded audio").await?,
            )),
        }
    }
}

impl OwnedAsrAudioInput {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Base64(audio) => audio.capacity(),
            Self::Bytes(audio) => audio.capacity(),
        }
    }

    fn decode(&self) -> Result<(Vec<f32>, u32)> {
        match self {
            Self::Base64(audio) => {
                let audio = base64_decode(audio)?;
                decode_audio_bytes_canonical(&audio)
            }
            Self::Bytes(audio) => decode_audio_bytes_canonical(audio),
        }
    }

    fn validate_source_size(&self) -> Result<()> {
        match self {
            Self::Base64(audio) => {
                validate_base64_audio_source_size(audio, MAX_AUDIO_SOURCE_BYTES)
            }
            Self::Bytes(audio) if audio.len() > MAX_AUDIO_SOURCE_BYTES => {
                Err(Error::InvalidInput(format!(
                    "ASR encoded audio is {} bytes, exceeding the {MAX_AUDIO_SOURCE_BYTES}-byte source limit",
                    audio.len()
                )))
            }
            Self::Bytes(_) => Ok(()),
        }
    }
}

/// IBM's plain-ASR decode budget (`ibm-granite/granite-speech`): 200 new
/// tokens per 30 s window, never less than one window. It covers punctuated
/// German at about 1.7 tokens per word.
const GRANITE_ASR_AUTO_TOKENS_PER_WINDOW: f64 = 200.0;
const GRANITE_ASR_AUTO_WINDOW_SECONDS: f64 = 30.0;
const GRANITE_ASR_AUTO_MAX_TOKENS: usize = 2048;
/// Plus's timestamp task writes `word [T:NNN]` (a tag is 5 tokens, plus
/// silence tags), about 24 tokens per second of speech measured with the
/// shipped tokenizer; budget 26 per second and a fixed reserve.
const GRANITE_TIMESTAMP_TOKENS_PER_SECOND: f64 = 26.0;
const GRANITE_TIMESTAMP_TOKEN_RESERVE: usize = 64;
const GRANITE_SAA_MIN_NEW_TOKENS: usize = 800;
/// 8 tokens a second for the longest chunk (510 s) plus the reserve:
/// speaker-attributed text runs ~3.3 tokens a second, so a dense nine-minute
/// chunk (plus v2610's context allows them) still has room.
const GRANITE_SAA_MAX_NEW_TOKENS: usize = 4_608;
const GRANITE_SAA_NEW_TOKENS_PER_SECOND: f32 = 8.0;
const GRANITE_SAA_NEW_TOKEN_RESERVE: usize = 256;
const GRANITE_SAA_TARGET_CHUNK_SECS: f32 = 240.0;
const GRANITE_SAA_HARD_MAX_CHUNK_SECS: f32 = 510.0;
/// Audio two consecutive chunks share: enough words for [`saa_seam`] to
/// align, and usually both speakers of a conversation.
const GRANITE_SAA_OVERLAP_SECS: f32 = 30.0;
const GRANITE_SAA_MIN_CHUNK_SECS: f32 = 30.0;
const GRANITE_SAA_SILENCE_SEARCH_SECS: f32 = 20.0;
const UNKNOWN_SAA_SPEAKER: &str = "UNKNOWN";
/// Concurrent realtime sessions on CPU and Metal, which is also the row count
/// of the realtime state arena a load allocates up front: Nemotron's is about
/// 37 MB of F32 per row (twice per session), so 16 sessions held 1.18 GB that
/// a desktop rarely uses (NEM-4).
const DEFAULT_ASR_REALTIME_MAX_SESSIONS: usize = 4;
/// Concurrent realtime sessions on CUDA: a serving GPU's arena of 16 rows is
/// a few percent of its memory, and the load's admission prices it.
const DEFAULT_CUDA_ASR_REALTIME_MAX_SESSIONS: usize = 16;
/// A whole working day of captions or a long meeting. Voxtral rolls its
/// stream over at a pause before its model ceiling and Nemotron's state is
/// bounded, so the lifetime only reaps a session nobody closes; it used to
/// force-close every meeting transcription at 10 minutes.
const DEFAULT_ASR_REALTIME_MAX_LIFETIME_SECS: u64 = 8 * 60 * 60;
/// A muted microphone that stops sending frames keeps its session this long;
/// 30 s reaped it in an ordinary pause in a conversation.
const DEFAULT_ASR_REALTIME_IDLE_TIMEOUT_SECS: u64 = 90;

/// Conservative decoder-prompt token reserve for the text surrounding the
/// audio placeholder (system block, role markers, task instructions). Chunk
/// planners subtract it from the effective invocation context before
/// budgeting audio tokens; per-invocation admission uses exact counts.
pub(crate) const ASR_TEXT_PROMPT_TOKEN_RESERVE: usize = 128;

/// Tokens one ASR decoder sequence needs for one full chunk (P2.5): its text
/// prompt, the audio tokens of `chunk_secs` at `audio_token_rate`, and
/// `decode_tokens` of output. No request decodes more than one chunk at a
/// time (longer input is chunked), so this, not the model's native context,
/// is the automatic per-sequence ceiling of every LLM-decoder ASR family.
pub(crate) fn asr_sequence_ceiling(
    text_prompt_tokens: usize,
    audio_token_rate: f32,
    chunk_secs: f32,
    decode_tokens: usize,
) -> usize {
    let audio_tokens = (chunk_secs.max(0.0) * audio_token_rate.max(0.0)).ceil() as usize;
    text_prompt_tokens
        .saturating_add(audio_tokens)
        .saturating_add(decode_tokens)
}

/// `(sequence, retained)` token bounds of an automatic ASR context: one
/// `sequence_ceiling` per sequence, and one per concurrent sequence in the
/// shared retained pool, never above the model's native context.
pub(crate) fn asr_automatic_context(
    native_tokens: usize,
    sequence_ceiling: usize,
    retained_rows: usize,
) -> (usize, usize) {
    let sequence = sequence_ceiling.min(native_tokens);
    let retained = sequence
        .saturating_mul(retained_rows.max(1))
        .min(native_tokens);
    (sequence, retained)
}

/// Smallest audio duration in seconds an invocation chunk planner will still
/// emit before declaring the invocation context infeasible.
pub(crate) const ASR_MIN_VIABLE_CHUNK_SECS: f32 = 4.0;

const ASR_MAX_SOLVED_CHUNK_SECS: f32 = 24.0 * 3600.0;

/// Outcome of fitting an audio duration to an invocation context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum InvocationAudioLimit {
    /// Even the smallest viable invocation cannot fit the context.
    Infeasible,
    /// Largest admissible audio duration in seconds.
    Limited(f32),
}

/// Largest audio duration whose invocation demand — text reserve, audio
/// tokens at the family's rate, and the route's decode budget — fits
/// `context_tokens`. Demand is monotone non-decreasing in duration, so the
/// boundary is found by exponential bracketing plus bisection.
pub(crate) fn solve_invocation_audio_limit(
    context_tokens: usize,
    audio_token_rate: f32,
    text_reserve_tokens: usize,
    decode_budget_for_duration: impl Fn(f32) -> usize,
) -> InvocationAudioLimit {
    if context_tokens == 0 || !audio_token_rate.is_finite() || audio_token_rate <= 0.0 {
        return InvocationAudioLimit::Infeasible;
    }
    let fits = |secs: f32| -> bool {
        text_reserve_tokens
            .saturating_add(invocation_audio_tokens(secs, audio_token_rate))
            .saturating_add(decode_budget_for_duration(secs))
            <= context_tokens
    };
    if !fits(ASR_MIN_VIABLE_CHUNK_SECS) {
        return InvocationAudioLimit::Infeasible;
    }
    let mut low = ASR_MIN_VIABLE_CHUNK_SECS;
    let mut high = ASR_MIN_VIABLE_CHUNK_SECS * 2.0;
    while fits(high) && high < ASR_MAX_SOLVED_CHUNK_SECS {
        low = high;
        high *= 2.0;
    }
    if fits(high) {
        return InvocationAudioLimit::Limited(high);
    }
    for _ in 0..40 {
        let middle = (low + high) / 2.0;
        if fits(middle) {
            low = middle;
        } else {
            high = middle;
        }
    }
    InvocationAudioLimit::Limited(low)
}

fn invocation_audio_tokens(audio_seconds: f32, audio_token_rate: f32) -> usize {
    (audio_seconds * audio_token_rate).ceil() as usize
}

/// Decode tokens `context_tokens` leaves after the prompt of an
/// `audio_seconds` invocation, under the demand model the chunk planner
/// solves with (text reserve plus audio tokens at the family's rate).
pub(crate) fn invocation_decode_room(
    context_tokens: usize,
    audio_token_rate: f32,
    audio_seconds: f32,
) -> usize {
    context_tokens.saturating_sub(ASR_TEXT_PROMPT_TOKEN_RESERVE.saturating_add(
        invocation_audio_tokens(audio_seconds.max(0.0), audio_token_rate),
    ))
}

/// Compose the family's audio-seconds hint with the invocation-context budget:
/// whichever is smaller governs chunk planning. Families whose audio never
/// reaches the decoder (`audio_token_rate == None`) keep their hint alone.
/// `Err` means the effective context cannot admit even a minimal invocation.
pub(crate) fn compose_invocation_audio_limit(
    audio_seconds_hint: Option<f32>,
    audio_token_rate: Option<f32>,
    effective_context: Option<usize>,
    decode_budget_for_duration: impl Fn(f32) -> usize,
) -> Result<Option<f32>> {
    let Some(context) = effective_context else {
        return Ok(audio_seconds_hint);
    };
    let Some(rate) = audio_token_rate else {
        return Ok(audio_seconds_hint);
    };
    match solve_invocation_audio_limit(
        context,
        rate,
        ASR_TEXT_PROMPT_TOKEN_RESERVE,
        &decode_budget_for_duration,
    ) {
        InvocationAudioLimit::Infeasible => Err(Error::InvalidInput(format!(
            "Effective invocation context ({context} tokens) cannot fit even a minimal {rate:.1}-token/second audio invocation; free memory or raise the configured max_sequence_length"
        ))),
        InvocationAudioLimit::Limited(secs) => Ok(match audio_seconds_hint {
            Some(hint) => Some(hint.min(secs)),
            None => Some(secs),
        }),
    }
}

/// The noise gate a realtime stream on `variant` feeds its model through:
/// one for model families that hallucinate on non-speech, unless
/// `IZWI_ASR_REALTIME_NOISE_GATE=0`. It delays transcripts by
/// [`izwi_vad::DEFAULT_NOISE_GATE_LOOKAHEAD_MS`].
fn realtime_noise_gate(variant: ModelVariant) -> Option<izwi_vad::NoiseGate> {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| crate::env::flag_or("IZWI_ASR_REALTIME_NOISE_GATE", true));
    (variant.family().hallucinates_on_non_speech() && enabled).then(|| {
        izwi_vad::NoiseGate::new(izwi_vad::NoiseGateConfig {
            endpoint: izwi_vad::EndpointConfig {
                snr_margin_db: izwi_vad::configured_snr_margin_db(),
                continue_snr_margin_db: izwi_vad::configured_snr_margin_db() * 0.5,
                ..izwi_vad::EndpointConfig::default()
            },
            ..izwi_vad::NoiseGateConfig::default()
        })
    })
}

fn resolve_asr_realtime_stream_variant(model_id: Option<&str>) -> Option<ModelVariant> {
    let variant = resolve_asr_model_variant(model_id);
    capability_row(CapabilityKind::RealtimeAsr, variant).map(|_| variant)
}

/// Check a request's language against what the routed ASR model was trained
/// on, so an unsupported language fails as invalid input before admission.
/// Only families with a published language list (Granite Speech) check it.
pub fn validate_asr_language(variant: ModelVariant, language: Option<&str>) -> Result<()> {
    match variant.family() {
        ModelFamily::GraniteSpeechAsr => {
            validate_granite_speech_request_language(variant, language)
        }
        ModelFamily::GraniteSpeech5Asr => {
            resolve_granite_speech5_source_language(language).map(|_| ())
        }
        _ => Ok(()),
    }
}

/// Reject a translation target or keyword list `variant` cannot honour
/// before the request is admitted. Only Granite Speech 5.0 translates and
/// takes structured keywords; its own validation checks the language pair
/// and the keyword limits against the rest of the request.
pub fn validate_asr_task(
    variant: ModelVariant,
    language: Option<&str>,
    prompt: Option<&str>,
    task: &AsrTaskOptions,
) -> Result<()> {
    if task.translate_to.is_some() && !variant.supports_asr_translation() {
        return Err(Error::InvalidInput(format!(
            "{variant} transcribes only; speech translation needs a model that supports it"
        )));
    }
    if !task.keywords.is_empty() && !variant.supports_asr_keyword_biasing() {
        return Err(Error::InvalidInput(format!(
            "{variant} does not take keywords; pass them in the prompt instead"
        )));
    }
    match variant.family() {
        ModelFamily::GraniteSpeech5Asr => {
            validate_granite_speech5_request(&GraniteSpeech5RequestOptions {
                language: language.map(str::to_string),
                prompt: prompt.map(str::to_string),
                translate_to: task.translate_to.clone(),
                keywords: task.keywords.clone(),
            })
        }
        _ => Ok(()),
    }
}

/// IBM's automatic Granite budget, `ceil(200 * max(1, secs / 30))` tokens.
/// Callers bound it by the request ceiling and the invocation context.
pub(crate) fn granite_auto_asr_max_tokens_for_duration(audio_seconds: f32) -> usize {
    let windows = if audio_seconds.is_finite() {
        (f64::from(audio_seconds) / GRANITE_ASR_AUTO_WINDOW_SECONDS).max(1.0)
    } else {
        1.0
    };
    (GRANITE_ASR_AUTO_TOKENS_PER_WINDOW * windows).ceil() as usize
}

/// Budget for plus's timestamp task, `ceil(26 * secs) + 64` tokens.
pub(crate) fn granite_timestamp_max_tokens_for_duration(audio_seconds: f32) -> usize {
    let seconds = if audio_seconds.is_finite() {
        f64::from(audio_seconds).max(0.0)
    } else {
        0.0
    };
    (GRANITE_TIMESTAMP_TOKENS_PER_SECOND * seconds).ceil() as usize
        + GRANITE_TIMESTAMP_TOKEN_RESERVE
}

/// Decode budget Granite plans an invocation of `audio_seconds` with: the
/// automatic duration budget of the task the request runs (plain ASR or
/// plus's word timestamps) under the request ceiling, or the caller's own
/// `max_tokens`. Chunk planning and per-chunk decoding share it, so a planned
/// chunk always leaves room for the budget it is decoded with.
pub(crate) fn granite_asr_planned_max_tokens(
    request: &EngineCoreRequest,
    audio_seconds: f32,
) -> usize {
    let ceiling = request.params.max_tokens.max(1);
    if !request.asr_auto_max_tokens {
        return ceiling;
    }
    let word_timestamps = request.model_variant.is_some_and(|variant| {
        granite_speech_request_plan(variant, request.asr_native_word_timestamps).task
            == GraniteSpeechTask::WordTimestamps
    });
    let budget = if word_timestamps {
        granite_timestamp_max_tokens_for_duration(audio_seconds)
    } else {
        granite_auto_asr_max_tokens_for_duration(audio_seconds)
    };
    budget.min(ceiling)
}

fn granite_auto_asr_token_ceiling(variant: ModelVariant) -> Option<usize> {
    // The executor already derives the exact per-chunk budget after its
    // admitted blocking audio decode. Reserving the production ceiling here
    // avoids decoding compressed public input synchronously on a Tokio worker
    // merely to construct the engine request.
    match variant.family() {
        ModelFamily::GraniteSpeechAsr => Some(GRANITE_ASR_AUTO_MAX_TOKENS),
        ModelFamily::GraniteSpeech5Asr => Some(GRANITE_SPEECH5_AUTO_MAX_TOKENS),
        _ => None,
    }
}

/// Decode budget Granite Speech 5.0 plans and decodes an invocation of
/// `audio_seconds` with: its automatic duration budget under the request
/// ceiling, or the caller's own `max_tokens`.
pub(crate) fn granite_speech5_asr_planned_max_tokens(
    request: &EngineCoreRequest,
    audio_seconds: f32,
) -> usize {
    let ceiling = request.params.max_tokens.max(1);
    if !request.asr_auto_max_tokens {
        return ceiling;
    }
    granite_speech5_auto_max_tokens_for_duration(audio_seconds).min(ceiling)
}

/// Transcribed audio duration. Preparation records it for every family that
/// decodes before admission; otherwise only the executor's report is known.
fn asr_transcription_duration_secs(output: &EngineOutput) -> f32 {
    output
        .input_audio_duration_secs
        .unwrap_or(output.audio.duration_secs)
}

/// The final transcript, cleaned exactly as the long-form assembler cleans
/// each chunk, so a file decoded in one window and a file merged from chunks
/// come back alike. Cleaning is idempotent, so merged text is unchanged.
fn asr_transcript_text(text: &str) -> String {
    clean_transcript_text(text, AsrLongFormConfig::default().max_repeated_chars)
}

/// Whether `variant` streams ASR on a native streaming contract. Its ASR
/// adapter's streaming mode decides; loaded adapters without one reject a
/// native contract and accept only transport progress from offline decode.
pub(super) fn asr_streams_natively(
    adapters: &RuntimeAdapterRegistry,
    variant: ModelVariant,
) -> bool {
    adapters
        .require(CapabilityKind::Asr, variant)
        .is_ok_and(|adapter| adapter.streaming_mode != StreamingMode::None)
}

/// The part of the final transcript a stream has not delivered, so the
/// deltas concatenate to it: all of it when nothing was streamed, the rest
/// after the streamed prefix (allowing for leading whitespace the cleanup
/// trimmed), and nothing when the cleanup rewrote text already sent.
fn unstreamed_transcript_tail<'a>(streamed: &str, text: &'a str) -> Option<&'a str> {
    text.strip_prefix(streamed)
        .or_else(|| text.strip_prefix(streamed.trim_start()))
        .filter(|tail| !tail.is_empty())
}

/// Transcript language: the translation target when the request translates,
/// the caller's hint when given, otherwise the language the model detected
/// and reported as `language` in its ASR diagnostics.
fn asr_response_language(
    requested: Option<&str>,
    translate_to: Option<&str>,
    diagnostics: Option<&serde_json::Value>,
) -> Option<String> {
    translate_to.or(requested).map(str::to_string).or_else(|| {
        diagnostics
            .and_then(|diagnostics| diagnostics.get("language"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
    })
}

/// Install the caller's ASR decode budget. An omitted budget is marked
/// automatic so duration-aware families (Granite, LFM2.5 Audio long-form) may
/// derive it per chunk, and Parakeet preparation may size it from the encoder
/// frame count, instead of applying the request default.
fn apply_asr_max_tokens(
    request: &mut EngineCoreRequest,
    variant: ModelVariant,
    max_tokens: Option<usize>,
) {
    if let Some(max_tokens) = max_tokens {
        request.params.max_tokens = max_tokens;
        return;
    }
    request.asr_auto_max_tokens = true;
    if let Some(auto_max_tokens) = granite_auto_asr_token_ceiling(variant)
        .or_else(|| vibevoice_auto_asr_token_ceiling(variant))
    {
        request.params.max_tokens = auto_max_tokens;
    }
}

/// VibeVoice-ASR's automatic budget is admitted at the reference ceiling and
/// narrowed to the decoded audio's duration by preparation and the executor.
fn vibevoice_auto_asr_token_ceiling(variant: ModelVariant) -> Option<usize> {
    use crate::models::architectures::vibevoice::asr::VIBEVOICE_ASR_AUTO_MAX_TOKENS;
    (variant.family() == ModelFamily::VibeVoiceAsr).then_some(VIBEVOICE_ASR_AUTO_MAX_TOKENS)
}

/// Decode budget a VibeVoice-ASR invocation of `audio_seconds` runs with: the
/// duration budget under the request ceiling when the caller omitted
/// `max_tokens`, otherwise the caller's own budget.
pub(crate) fn vibevoice_asr_planned_max_tokens(
    request: &EngineCoreRequest,
    audio_seconds: f32,
) -> usize {
    use crate::models::architectures::vibevoice::asr::vibevoice_auto_asr_max_tokens_for_duration;
    let ceiling = request.params.max_tokens.max(1);
    if request.asr_auto_max_tokens {
        vibevoice_auto_asr_max_tokens_for_duration(audio_seconds).min(ceiling)
    } else {
        ceiling
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RealtimeAsrSessionLimits {
    max_sessions: usize,
    max_lifetime: Duration,
    idle_timeout: Duration,
}

#[derive(Clone)]
pub(super) struct RealtimeAsrSessionPolicy {
    limits: RealtimeAsrSessionLimits,
    permits: Arc<Semaphore>,
    /// Direct-path pushes waiting to run with other streams' pushes.
    pushes: Arc<RealtimePushBatcher<RealtimePushRow>>,
}

pub struct RuntimeAsrRealtimeStream {
    variant: ModelVariant,
    /// Silences non-speech before it reaches a model that hallucinates on it
    /// (see [`realtime_noise_gate`]).
    noise_gate: Option<izwi_vad::NoiseGate>,
    resources: Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
    operation_gate: Arc<tokio::sync::Mutex<()>>,
    max_samples: usize,
    committed_samples: usize,
    engine_sample_rate: Option<u32>,
    engine_text: String,
    engine_chunk_index: usize,
    segment: RealtimeEngineSegment,
}

#[derive(Debug, Clone)]
pub struct RuntimeAsrRealtimeEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

fn host_input_observation(input_bytes: usize) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::host(
        u64::try_from(input_bytes)
            .map_err(|_| Error::InvalidInput("runtime input size exceeds u64".to_string()))?,
    ))
}

fn retained_host_observation(parts: &[usize]) -> Result<JobResourceObservation> {
    let bytes = parts.iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or_else(|| Error::Overloaded("retained ASR storage overflowed".to_string()))
    })?;
    host_input_observation(bytes)
}

fn add_retained_host_bytes(
    mut observation: JobResourceObservation,
    retained_bytes: usize,
) -> Result<JobResourceObservation> {
    let retained_bytes = u64::try_from(retained_bytes)
        .map_err(|_| Error::Overloaded("retained ASR storage exceeds u64".to_string()))?;
    observation.host_bytes = observation
        .host_bytes
        .checked_add(retained_bytes)
        .ok_or_else(|| Error::Overloaded("retained ASR storage overflowed".to_string()))?;
    Ok(observation)
}

fn decoded_audio_observation(
    input_bytes: usize,
    sample_capacity: usize,
) -> Result<JobResourceObservation> {
    let sample_bytes = sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(input_bytes))
        .ok_or_else(|| Error::Overloaded("decoded audio storage overflowed".to_string()))?;
    host_input_observation(sample_bytes)
}

fn decoded_audio_with_scratch_observation(
    input_bytes: usize,
    sample_count: usize,
    scratch_sample_count: usize,
) -> Result<JobResourceObservation> {
    let samples = sample_count
        .checked_add(scratch_sample_count)
        .ok_or_else(|| Error::Overloaded("ASR sample storage overflowed".to_string()))?;
    decoded_audio_observation(input_bytes, samples)
}

impl RuntimeService {
    /// Force alignment remains a specialized operation not expressed by the
    /// generic engine output type.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, None)
            .await
    }

    pub async fn force_align_with_model(
        &self,
        audio_base64: &str,
        reference_text: &str,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, model_id)
            .await
    }

    pub async fn force_align_with_model_and_language(
        &self,
        audio_base64: &str,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_input(
            AsrAudioInput::Base64(audio_base64),
            ForcedAlignmentText::Whole(reference_text),
            language,
            model_id,
        )
        .await?
        .into_words_or_error()
    }

    pub async fn force_align_bytes_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_input(
            AsrAudioInput::Bytes(audio_bytes),
            ForcedAlignmentText::Whole(reference_text),
            language,
            model_id,
        )
        .await?
        .into_words_or_error()
    }

    /// Align one transcript against the whole audio. Audio longer than the
    /// aligner's window is aligned in sub-windows (see [`LongTextPlanner`]);
    /// each reports where its timings came from.
    pub async fn force_align_bytes_windowed_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<WindowedAlignment> {
        self.force_align_input(
            AsrAudioInput::Bytes(audio_bytes),
            ForcedAlignmentText::Whole(reference_text),
            language,
            model_id,
        )
        .await
    }

    /// Align each window's text against its own range of the audio, one
    /// coordinator stage per window, with times rebased to the whole file.
    /// A window the aligner cannot align keeps interval timings and reports
    /// it in its source; only admission, decode or load failures error.
    pub async fn force_align_windows_bytes_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        windows: Vec<AlignmentWindow>,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<WindowedAlignment> {
        self.force_align_input(
            AsrAudioInput::Bytes(audio_bytes),
            ForcedAlignmentText::Windows(windows),
            language,
            model_id,
        )
        .await
    }

    async fn force_align_input(
        &self,
        audio_input: AsrAudioInput<'_>,
        text: ForcedAlignmentText<'_>,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<WindowedAlignment> {
        let variant = resolve_forced_aligner_variant(model_id)?;
        if audio_input.input_bytes() == 0 {
            return Err(Error::InvalidInput(
                "forced alignment request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        if text.is_blank() {
            return Err(Error::InvalidInput(
                "Forced alignment request missing transcript".to_string(),
            ));
        }
        self.observe_broker_capability_request(
            CapabilityKind::ForcedAlignment,
            Some(variant),
            false,
        )?;
        let input_bytes = audio_input
            .input_bytes()
            .checked_add(text.input_bytes())
            .and_then(|bytes| bytes.checked_add(language.map(str::len).unwrap_or_default()))
            .ok_or_else(|| Error::Overloaded("forced alignment input overflowed".to_string()))?;
        let context = RuntimeRequestContext::default();
        let spec = self.coordinator_job_for_audio_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Atomic,
            context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;
        let audio_input = audio_input.into_owned_for_job(&job).await?;
        let windows = match text {
            ForcedAlignmentText::Whole(text) => ForcedAlignmentOwnedText::Whole(
                copy_preparation_string(&job, text, "forced alignment transcript").await?,
            ),
            ForcedAlignmentText::Windows(windows) => ForcedAlignmentOwnedText::Windows(windows),
        };
        let language =
            copy_optional_preparation_string(&job, language, "forced alignment language").await?;
        let owned_metadata_bytes = windows
            .retained_bytes()
            .checked_add(language.as_ref().map(String::capacity).unwrap_or_default())
            .ok_or_else(|| {
                Error::Overloaded("forced alignment metadata storage overflowed".to_string())
            })?;
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            audio_input.retained_bytes(),
            owned_metadata_bytes,
        ])?)?;
        let (residency_lease, execution_contract) = self
            .load_capability_for_job(
                &job,
                variant,
                CapabilityKind::ForcedAlignment,
                false,
                ExecutionTargetKind::BatchRunner,
            )
            .await?;
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        let observation_job = job.clone();
        let (samples, sample_rate) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let retained_audio_bytes = audio_input.retained_bytes();
                let (samples, sample_rate) = audio_input.decode()?;
                let retained_bytes =
                    input_bytes
                        .checked_add(owned_metadata_bytes)
                        .ok_or_else(|| {
                            Error::Overloaded(
                                "forced alignment retained storage overflowed".to_string(),
                            )
                        })?;
                let steady_usage = decoded_audio_observation(retained_bytes, samples.capacity())?;
                observation_job.record_materialized_usage(add_retained_host_bytes(
                    steady_usage,
                    retained_audio_bytes,
                )?)?;
                observation_job.prepare_materialized_release(steady_usage)?;
                drop(audio_input);
                Ok((Arc::<[f32]>::from(samples), sample_rate))
            })
            .await?;
        let windows = match windows {
            ForcedAlignmentOwnedText::Whole(text) => vec![SampleWindow::new(
                0,
                samples.len(),
                text,
                language.as_deref(),
            )],
            ForcedAlignmentOwnedText::Windows(windows) => {
                sample_windows(windows, sample_rate, samples.len(), language.as_deref())?
            }
        };
        let duration_ms = samples_to_ms(samples.len(), sample_rate);
        let windows = self
            .align_sample_windows(
                &job,
                execution_contract,
                model,
                samples,
                sample_rate,
                windows,
                "asr.forced_alignment",
            )
            .await?;
        drop(residency_lease);
        Ok(WindowedAlignment {
            windows,
            duration_ms,
        })
    }

    /// Run each window as its own aligner stage, so other jobs interleave
    /// between windows. A window longer than the aligner's limit is split by
    /// [`LongTextPlanner`] into sub-windows, each its own stage. Shared by the
    /// alignment API and the diarization pipeline; the caller holds the
    /// aligner's residency lease. The windows carry their word units.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn align_sample_windows(
        &self,
        job: &JobLease,
        contract: LoadedExecutionContract,
        model: Arc<NativeAsrModel>,
        samples: Arc<[f32]>,
        sample_rate: u32,
        windows: Vec<SampleWindow>,
        stage_kind: &str,
    ) -> Result<Vec<AlignedWindow>> {
        let max_window = model
            .max_audio_seconds_hint()
            .filter(|seconds| seconds.is_finite() && *seconds > 0.0)
            .map(|seconds| (f64::from(seconds) * f64::from(sample_rate)).floor() as usize);
        let mut aligned = Vec::with_capacity(windows.len());
        for window in windows {
            let Some(max_window) = max_window
                .filter(|max| window.end_sample.saturating_sub(window.start_sample) > *max)
            else {
                aligned.push(
                    self.align_window_stage(
                        job,
                        &contract,
                        &model,
                        &samples,
                        sample_rate,
                        window,
                        stage_kind,
                        true,
                    )
                    .await?,
                );
                continue;
            };
            // The energy scan covers the whole window: run it off the
            // async workers.
            let planner_samples = samples.clone();
            let mut planner = self
                .coordinator
                .run_host_blocking_stage(job, move || {
                    Ok(LongTextPlanner::new(
                        &planner_samples,
                        sample_rate,
                        &window,
                        max_window,
                    ))
                })
                .await?;
            while let Some(planned) = planner.next_window(&samples) {
                let sub_window = self
                    .align_window_stage(
                        job,
                        &contract,
                        &model,
                        &samples,
                        sample_rate,
                        planned.window.clone(),
                        stage_kind,
                        false,
                    )
                    .await?;
                aligned.push(planner.absorb(&planned, sub_window));
            }
        }
        Ok(aligned)
    }

    #[allow(clippy::too_many_arguments)]
    async fn align_window_stage(
        &self,
        job: &JobLease,
        contract: &LoadedExecutionContract,
        model: &Arc<NativeAsrModel>,
        samples: &Arc<[f32]>,
        sample_rate: u32,
        window: SampleWindow,
        stage_kind: &str,
        check_plausibility: bool,
    ) -> Result<AlignedWindow> {
        // The aligner adapter is stateless, so the window's forward working
        // set is reserved here, for this stage only (ADM-5).
        let workspace = forced_alignment_window_workspace(
            contract.metadata.model_variant,
            self.backend_router.context().backend_kind,
            &window,
            sample_rate,
        )?;
        let lease = self
            .coordinator
            .reserve_stage_workspace(job, stage_kind, workspace)?;
        let model = model.clone();
        let samples = samples.clone();
        self.coordinator
            .run_loaded_blocking_stage(
                job,
                contract.clone(),
                WorkUnit::AtomicJob {
                    kind: stage_kind.to_string(),
                },
                move || {
                    align_under_stage_lease(&lease, workspace, || {
                        align_window_checked(
                            &samples,
                            sample_rate,
                            &window,
                            check_plausibility,
                            |audio, units| model.force_align_units(audio, sample_rate, units),
                        )
                    })
                },
            )
            .await
    }
}

/// The transcript of a forced alignment request: one text for the whole
/// audio, or the text of each window.
enum ForcedAlignmentText<'a> {
    Whole(&'a str),
    Windows(Vec<AlignmentWindow>),
}

impl ForcedAlignmentText<'_> {
    fn is_blank(&self) -> bool {
        match self {
            Self::Whole(text) => text.trim().is_empty(),
            Self::Windows(windows) => windows.iter().all(|window| window.text.trim().is_empty()),
        }
    }

    fn input_bytes(&self) -> usize {
        match self {
            Self::Whole(text) => text.len(),
            Self::Windows(windows) => windows.iter().map(|window| window.text.len()).sum(),
        }
    }
}

enum ForcedAlignmentOwnedText {
    Whole(String),
    Windows(Vec<AlignmentWindow>),
}

impl ForcedAlignmentOwnedText {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Whole(text) => text.capacity(),
            Self::Windows(windows) => windows
                .iter()
                .map(|window| window.text.capacity() + std::mem::size_of::<AlignmentWindow>())
                .sum(),
        }
    }
}

pub(crate) fn resolve_forced_aligner_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::Qwen3ForcedAligner06B,
    };

    if !variant.is_forced_aligner() {
        return Err(Error::InvalidInput(format!(
            "Model {} is not a forced aligner model",
            variant.dir_name()
        )));
    }

    Ok(variant)
}

#[cfg(all(test, unix))]
#[path = "asr_tests.rs"]
mod tests;
