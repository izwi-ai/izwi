//! ASR runtime methods routed through the unified core engine.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use crate::backends::state::{PhysicalStateSequenceId, PhysicalStateTransactionId};
use crate::backends::BackendKind;
use crate::catalog::{parse_model_variant, resolve_asr_model_variant, ModelFamily};
use crate::config::RealtimeAsrSessionConfig;
use crate::engine::{
    AsrProgress, AsrProgressPhase, EngineCoreRequest, EngineOutput, RealtimeAsrSessionHandle,
    ResourceAmount, ResourceVector, RetainedTensorStateRuntimeV2, StreamingOutput, TaskType,
    WorkCost, WorkUnit,
};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::models::architectures::granite_speech::asr::{
    granite_speech_request_plan, parse_granite_speech_output,
    validate_granite_speech_request_language, GraniteSpeechTask,
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
const GRANITE_SAA_MAX_NEW_TOKENS: usize = 2_500;
const GRANITE_SAA_NEW_TOKENS_PER_SECOND: f32 = 8.0;
const GRANITE_SAA_NEW_TOKEN_RESERVE: usize = 256;
const GRANITE_SAA_TARGET_CHUNK_SECS: f32 = 240.0;
const GRANITE_SAA_HARD_MAX_CHUNK_SECS: f32 = 510.0;
const GRANITE_SAA_OVERLAP_SECS: f32 = 12.0;
const GRANITE_SAA_MIN_CHUNK_SECS: f32 = 30.0;
const GRANITE_SAA_SILENCE_SEARCH_SECS: f32 = 12.0;
const GRANITE_SAA_PREFIX_MAX_TURNS: usize = 12;
const GRANITE_SAA_PREFIX_MAX_CHARS: usize = 6_000;
const GRANITE_SAA_MIN_OVERLAP_WORDS: usize = 4;
const GRANITE_SAA_MAX_OVERLAP_WORDS: usize = 80;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraniteSaaPrefixMode {
    None,
    FullTranscript,
}

impl GraniteSaaPrefixMode {
    /// `IZWI_GRANITE_SAA_PREFIX_MODE`, read once per process.
    fn from_env() -> Self {
        static MODE: crate::env::ReadOnce<GraniteSaaPrefixMode> =
            crate::env::ReadOnce::new(GraniteSaaPrefixMode::read_env);
        MODE.get()
    }

    fn read_env() -> Self {
        match std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE")
            .ok()
            .map(|raw| raw.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("full") | Some("full_transcript") | Some("legacy") => Self::FullTranscript,
            _ => Self::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FullTranscript => "full_transcript",
        }
    }
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
    (variant.family() == ModelFamily::GraniteSpeechAsr).then_some(GRANITE_ASR_AUTO_MAX_TOKENS)
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

/// Transcript language: the caller's hint when given, otherwise the language
/// the model detected and reported as `language` in its ASR diagnostics.
fn asr_response_language(
    requested: Option<&str>,
    diagnostics: Option<&serde_json::Value>,
) -> Option<String> {
    requested.map(str::to_string).or_else(|| {
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

impl RealtimeAsrSessionLimits {
    /// The engine config's limits, else the environment's, else the
    /// defaults for `backend`.
    fn resolve(config: RealtimeAsrSessionConfig, backend: BackendKind) -> Result<Self> {
        fn configured<T: Default + PartialEq>(value: Option<T>, field: &str) -> Result<Option<T>> {
            match value {
                Some(value) if value == T::default() => Err(Error::ConfigError(format!(
                    "asr_realtime.{field} must be positive"
                ))),
                value => Ok(value),
            }
        }
        let default_sessions = match backend {
            BackendKind::Cuda => DEFAULT_CUDA_ASR_REALTIME_MAX_SESSIONS,
            BackendKind::Cpu | BackendKind::Metal => DEFAULT_ASR_REALTIME_MAX_SESSIONS,
        };
        let max_sessions = match configured(config.max_sessions, "max_sessions")? {
            Some(sessions) => sessions,
            None => positive_usize_env_or_default(ENV_ASR_REALTIME_MAX_SESSIONS, default_sessions)?,
        };
        let max_lifetime_secs = match configured(config.max_lifetime_secs, "max_lifetime_secs")? {
            Some(secs) => secs,
            None => positive_u64_env_or_default(
                ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
                DEFAULT_ASR_REALTIME_MAX_LIFETIME_SECS,
            )?,
        };
        let idle_timeout_secs = match configured(config.idle_timeout_secs, "idle_timeout_secs")? {
            Some(secs) => secs,
            None => positive_u64_env_or_default(
                ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS,
                DEFAULT_ASR_REALTIME_IDLE_TIMEOUT_SECS,
            )?,
        };
        Ok(Self {
            max_sessions,
            max_lifetime: Duration::from_secs(max_lifetime_secs),
            idle_timeout: Duration::from_secs(idle_timeout_secs),
        })
    }
}

#[derive(Clone)]
pub(super) struct RealtimeAsrSessionPolicy {
    limits: RealtimeAsrSessionLimits,
    permits: Arc<Semaphore>,
}

impl RealtimeAsrSessionPolicy {
    pub(super) fn resolve(config: RealtimeAsrSessionConfig, backend: BackendKind) -> Result<Self> {
        Self::new(RealtimeAsrSessionLimits::resolve(config, backend)?)
    }

    fn new(limits: RealtimeAsrSessionLimits) -> Result<Self> {
        if limits.max_sessions > Semaphore::MAX_PERMITS {
            return Err(Error::ConfigError(format!(
                "{ENV_ASR_REALTIME_MAX_SESSIONS} cannot exceed {}",
                Semaphore::MAX_PERMITS
            )));
        }
        if Instant::now().checked_add(limits.max_lifetime).is_none() {
            return Err(Error::ConfigError(format!(
                "{ENV_ASR_REALTIME_MAX_LIFETIME_SECS} exceeds the platform clock range"
            )));
        }
        Ok(Self {
            permits: Arc::new(Semaphore::new(limits.max_sessions)),
            limits,
        })
    }

    pub(super) fn retained_sequence_capacity(&self) -> Result<u32> {
        u32::try_from(self.limits.max_sessions).map_err(|_| {
            Error::ConfigError(
                "realtime ASR session quota exceeds physical sequence capacity".into(),
            )
        })
    }

    fn try_acquire(&self) -> Result<Arc<RealtimeAsrSessionLease>> {
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Overloaded("realtime ASR session quota is full".to_string()))?;
        Ok(Arc::new(RealtimeAsrSessionLease { _permit: permit }))
    }
}

struct RealtimeAsrSessionLease {
    _permit: OwnedSemaphorePermit,
}

fn positive_usize_env_or_default(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

fn positive_u64_env_or_default(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

struct RealtimeAsrPhysicalSession {
    runtime: Arc<RetainedTensorStateRuntimeV2>,
    sequence: PhysicalStateSequenceId,
    active_transaction: Option<PhysicalStateTransactionId>,
    revision: u64,
}

impl RealtimeAsrPhysicalSession {
    fn new(runtime: Arc<RetainedTensorStateRuntimeV2>) -> Result<Self> {
        let sequence = runtime.register_sequence()?;
        Ok(Self {
            runtime,
            sequence,
            active_transaction: None,
            revision: 0,
        })
    }

    fn begin(&mut self) -> Result<PhysicalStateTransactionId> {
        if self.active_transaction.is_some() {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction is already active".into(),
            ));
        }
        let transaction = self.runtime.begin_transaction(self.sequence)?;
        self.active_transaction = Some(transaction);
        Ok(transaction)
    }

    fn commit(&mut self, transaction: PhysicalStateTransactionId, revision: u64) -> Result<()> {
        if self.active_transaction != Some(transaction) {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction identity changed".into(),
            ));
        }
        self.runtime.commit_transaction(transaction, revision)?;
        self.active_transaction = None;
        self.revision = revision;
        Ok(())
    }

    fn abort(&mut self) {
        if let Some(transaction) = self.active_transaction.take() {
            let _ = self.runtime.abort_transaction(transaction);
        }
    }
}

impl Drop for RealtimeAsrPhysicalSession {
    fn drop(&mut self) {
        self.abort();
        if let Err(error) = self.runtime.release_sequence(self.sequence) {
            tracing::error!(
                sequence = self.sequence.get(),
                error = %error,
                "failed to release realtime ASR physical sequence"
            );
        }
    }
}

struct RuntimeAsrRealtimeState {
    native: NativeAsrRealtimeState,
    physical: RealtimeAsrPhysicalSession,
}

impl RuntimeAsrRealtimeState {
    fn new(
        model: &NativeAsrModel,
        native: NativeAsrRealtimeState,
        runtime: Arc<RetainedTensorStateRuntimeV2>,
    ) -> Result<Self> {
        let mut owner = Self {
            native,
            physical: RealtimeAsrPhysicalSession::new(runtime)?,
        };
        let transaction = owner.physical.begin()?;
        let revision = 1;
        let result = model.stage_realtime_physical_state(
            &mut owner.native,
            &owner.physical.runtime,
            transaction,
            revision,
        );
        if let Err(error) = result {
            owner.physical.abort();
            return Err(error);
        }
        if let Err(error) = owner.physical.commit(transaction, revision) {
            owner.physical.abort();
            return Err(error);
        }
        Ok(owner)
    }

    fn transact<T>(
        &mut self,
        model: &NativeAsrModel,
        operation: impl FnOnce(&mut NativeAsrRealtimeState) -> Result<T>,
    ) -> Result<T> {
        let checkpoint = match &self.native {
            NativeAsrRealtimeState::Nemotron(state) => {
                NativeAsrRealtimeState::Nemotron(state.clone())
            }
        };
        let transaction = self.physical.begin()?;
        let result = (|| {
            model.hydrate_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
            )?;
            let output = operation(&mut self.native)?;
            let revision = self.physical.revision.checked_add(1).ok_or_else(|| {
                Error::InferenceError("realtime ASR physical revision overflowed".into())
            })?;
            model.stage_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
                revision,
            )?;
            self.physical.commit(transaction, revision)?;
            Ok(output)
        })();
        if result.is_err() {
            self.native = checkpoint;
            if let Err(error) = model.clear_realtime_tensor_handles(&mut self.native) {
                tracing::error!(
                    error = %error,
                    "failed to drain Nemotron tensor handles after aborted operation"
                );
            }
            self.physical.abort();
        }
        result
    }
}

struct RuntimeAsrRealtimeResources {
    model: Option<Arc<NativeAsrModel>>,
    state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    execution_contract: Option<LoadedExecutionContract>,
    residency_lease: Option<ModelResidencyLease>,
    job: Option<JobLease>,
    session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    engine_session: Option<RuntimeEngineRealtimeSession>,
    voxtral_model: Option<VoxtralModelLease>,
    absolute_deadline: Instant,
    idle_timeout: Duration,
    last_activity: Instant,
    active_operations: usize,
    closed: bool,
    timeout_reason: Option<&'static str>,
}

impl RuntimeAsrRealtimeResources {
    fn idle_deadline(&self) -> Instant {
        self.last_activity
            .checked_add(self.idle_timeout)
            .unwrap_or(self.absolute_deadline)
            .min(self.absolute_deadline)
    }

    fn next_watchdog_deadline(&self) -> Option<Instant> {
        if self.closed {
            None
        } else if self.active_operations > 0 {
            Some(self.absolute_deadline)
        } else {
            Some(self.idle_deadline())
        }
    }

    fn expiration(&self, now: Instant) -> Option<&'static str> {
        realtime_asr_session_expiration(
            now,
            self.absolute_deadline,
            self.idle_deadline(),
            self.active_operations,
        )
    }

    fn detach(&mut self) -> Option<RealtimeAsrDetachedResources> {
        if self.closed {
            return None;
        }
        self.closed = true;
        Some(RealtimeAsrDetachedResources {
            state: self.state.take(),
            model: self.model.take(),
            execution_contract: self.execution_contract.take(),
            residency_lease: self.residency_lease.take(),
            job: self.job.take(),
            session_lease: self.session_lease.take(),
            engine_session: self.engine_session.take(),
            voxtral_model: self.voxtral_model.take(),
        })
    }

    fn detach_expired(&mut self, reason: &'static str) -> Option<RealtimeAsrDetachedResources> {
        self.timeout_reason = Some(reason);
        self.detach()
    }

    fn closed_error(&self) -> Error {
        match self.timeout_reason {
            Some(reason) => Error::Timeout(reason.to_string()),
            None => Error::InvalidInput("realtime ASR stream is closed".to_string()),
        }
    }
}

struct RealtimeAsrDetachedResources {
    state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    model: Option<Arc<NativeAsrModel>>,
    execution_contract: Option<LoadedExecutionContract>,
    residency_lease: Option<ModelResidencyLease>,
    job: Option<JobLease>,
    session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    engine_session: Option<RuntimeEngineRealtimeSession>,
    voxtral_model: Option<VoxtralModelLease>,
}

impl RealtimeAsrDetachedResources {
    fn release(mut self) {
        // Physical allocations must disappear before their immutable
        // authorization and quota are released for another session.
        self.state.take();
        self.model.take();
        self.execution_contract.take();
        self.residency_lease.take();
        self.job.take();
        self.session_lease.take();
        self.engine_session.take();
        self.voxtral_model.take();
    }
}

/// Fail-closed owner for detached Runtime and Engine authorities. If the
/// asynchronous cleanup future is cancelled or its runtime is torn down,
/// dropping this guard deliberately retains every authority. Only an exact
/// Engine cleanup confirmation may disarm it.
struct RealtimeAsrCleanupGuard {
    cleanup: Option<RealtimeAsrDetachedResources>,
    engine_session: Option<RuntimeEngineRealtimeSession>,
    disarmed: bool,
}

impl RealtimeAsrCleanupGuard {
    fn new(
        cleanup: RealtimeAsrDetachedResources,
        engine_session: RuntimeEngineRealtimeSession,
    ) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: Some(engine_session),
            disarmed: false,
        }
    }

    fn engine_session(&self) -> &RuntimeEngineRealtimeSession {
        self.engine_session
            .as_ref()
            .expect("armed realtime cleanup guard must retain its Engine session")
    }

    #[cfg(test)]
    fn for_test(cleanup: RealtimeAsrDetachedResources) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: None,
            disarmed: false,
        }
    }

    fn release_confirmed(mut self) {
        self.disarmed = true;
        self.engine_session.take();
        if let Some(cleanup) = self.cleanup.take() {
            cleanup.release();
        }
    }
}

impl Drop for RealtimeAsrCleanupGuard {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        if let Some(engine_session) = self.engine_session.take() {
            std::mem::forget(engine_session);
        }
        if let Some(cleanup) = self.cleanup.take() {
            std::mem::forget(cleanup);
        }
    }
}

fn schedule_realtime_asr_cleanup(cleanup: Option<RealtimeAsrDetachedResources>) {
    let Some(mut cleanup) = cleanup else {
        return;
    };
    if let Some(engine_session) = cleanup.engine_session.take() {
        let guard = RealtimeAsrCleanupGuard::new(cleanup, engine_session);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                // Failures are logged and retain the authorities.
                let _ = confirm_realtime_asr_engine_cleanup(guard).await;
            });
            return;
        }
        tracing::error!(
            request_id = %guard.engine_session().handle.request_id(),
            "cannot asynchronously abort Engine realtime ASR session without a Tokio runtime"
        );
        drop(guard);
        return;
    }
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(move || cleanup.release());
    } else {
        cleanup.release();
    }
}

/// Releases a detached Engine session's authorities only after the Engine
/// confirms its cleanup. Any failure, or dropping this future, retains them.
async fn confirm_realtime_asr_engine_cleanup(guard: RealtimeAsrCleanupGuard) -> Result<()> {
    let cleanup_receipt = match guard
        .engine_session()
        .engine
        .cleanup_realtime_asr_session(&guard.engine_session().handle)
        .await
    {
        Ok(receipt) => receipt,
        Err(error) => {
            tracing::error!(
                request_id = %guard.engine_session().handle.request_id(),
                error = %error,
                "Engine realtime ASR cleanup receipt could not be created; retaining authorities"
            );
            return Err(error);
        }
    };
    if let Err(error) = cleanup_receipt.confirmed().await {
        tracing::error!(
            request_id = %guard.engine_session().handle.request_id(),
            error = %error,
            "Engine realtime ASR cleanup was not confirmed; retaining authorities"
        );
        return Err(error);
    }
    tokio::task::spawn_blocking(move || guard.release_confirmed())
        .await
        .map_err(|error| {
            Error::InferenceError(format!("realtime ASR cleanup release failed: {error}"))
        })
}

/// Like [`schedule_realtime_asr_cleanup`], but completes before returning,
/// so a rollover never holds two segments' physical state at once.
async fn release_realtime_asr_cleanup(cleanup: Option<RealtimeAsrDetachedResources>) -> Result<()> {
    let Some(mut cleanup) = cleanup else {
        return Ok(());
    };
    match cleanup.engine_session.take() {
        Some(engine_session) => {
            confirm_realtime_asr_engine_cleanup(RealtimeAsrCleanupGuard::new(
                cleanup,
                engine_session,
            ))
            .await
        }
        None => tokio::task::spawn_blocking(move || cleanup.release())
            .await
            .map_err(|error| {
                Error::InferenceError(format!("realtime ASR cleanup release failed: {error}"))
            }),
    }
}

struct RealtimeAsrOperationHandles {
    model: Arc<NativeAsrModel>,
    state: Arc<StdMutex<RuntimeAsrRealtimeState>>,
    execution_contract: LoadedExecutionContract,
    job: JobLease,
    session_lease: Arc<RealtimeAsrSessionLease>,
    _guard: RealtimeAsrOperationGuard,
}

#[derive(Clone)]
struct RuntimeEngineRealtimeSession {
    engine: Arc<crate::engine::Engine>,
    handle: RealtimeAsrSessionHandle,
    /// The Engine path serves Voxtral only: Nemotron's retained tensor state
    /// runs on the direct path (`realtime_asr_execution_path`).
    model: VoxtralModelLease,
    retained_metadata_bytes: u64,
    max_output_steps: usize,
    max_cache_append: usize,
}

/// Where a realtime session executes, from the capability table. Voxtral
/// publishes a paged managed state runtime, which the Engine's realtime
/// session admission requires; Nemotron publishes a retained tensor state
/// runtime and runs on the direct native streaming path (routing it through
/// Engine admission fails closed with "requires a physical managed state
/// runtime"). A model without a realtime row takes the direct path.
fn realtime_asr_execution_path(variant: ModelVariant) -> RealtimeAsrPath {
    capability_row(CapabilityKind::RealtimeAsr, variant)
        .and_then(|row| row.realtime_path)
        .unwrap_or(RealtimeAsrPath::Direct)
}

struct RealtimeAsrEngineOperationHandles {
    session: RuntimeEngineRealtimeSession,
    job: JobLease,
    session_lease: Arc<RealtimeAsrSessionLease>,
    _guard: RealtimeAsrOperationGuard,
}

struct RealtimeAsrOperationGuard {
    resources: Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
}

impl Drop for RealtimeAsrOperationGuard {
    fn drop(&mut self) {
        if let Ok(mut resources) = self.resources.lock() {
            resources.active_operations = resources.active_operations.saturating_sub(1);
            if !resources.closed {
                resources.last_activity = Instant::now();
            }
        }
        self.activity.notify_one();
    }
}

pub struct RuntimeAsrRealtimeStream {
    variant: ModelVariant,
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

/// An Engine realtime session prepares at most `max_samples` of source
/// audio (Voxtral re-encodes its whole prefix on every push), so a longer
/// stream continues in a fresh session. A segment looks for a pause in its
/// final fifth so the boundary falls between words; it rolls over at the
/// hard ceiling only when the speaker never pauses.
const REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR: usize = 5;
/// Sustained quiet audio that counts as a pause between words.
const REALTIME_SEGMENT_PAUSE_MS: u64 = 200;
/// Chunk RMS below which pushed audio counts as quiet (about -40 dBFS).
const REALTIME_SEGMENT_QUIET_RMS: f64 = 0.01;

/// What an Engine realtime stream carries from one session to the next.
#[derive(Debug, Default)]
struct RealtimeEngineSegment {
    language: Option<String>,
    prompt: Option<String>,
    trailing_quiet_samples: usize,
    separator_pending: bool,
}

impl RealtimeEngineSegment {
    fn new(language: Option<&str>, prompt: Option<&str>) -> Self {
        Self {
            language: language.map(str::to_string),
            prompt: prompt.map(str::to_string),
            ..Self::default()
        }
    }
}

fn realtime_chunk_is_quiet(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let energy: f64 = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum();
    (energy / samples.len() as f64).sqrt() < REALTIME_SEGMENT_QUIET_RMS
}

fn realtime_trailing_quiet_samples(previous: usize, chunk: &[f32]) -> usize {
    if realtime_chunk_is_quiet(chunk) {
        previous.saturating_add(chunk.len())
    } else {
        0
    }
}

/// Whether an Engine realtime stream must start a new session before it
/// accepts `incoming` more samples.
fn realtime_segment_rollover_due(
    committed_samples: usize,
    incoming_samples: usize,
    trailing_quiet_samples: usize,
    max_samples: usize,
    sample_rate: u32,
) -> bool {
    if committed_samples == 0 {
        return false;
    }
    if committed_samples.saturating_add(incoming_samples) > max_samples {
        return true;
    }
    let soft_start = max_samples - max_samples / REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR;
    let pause_samples = usize::try_from(u64::from(sample_rate) * REALTIME_SEGMENT_PAUSE_MS / 1_000)
        .unwrap_or(usize::MAX)
        .max(1);
    committed_samples >= soft_start && trailing_quiet_samples >= pause_samples
}

impl RuntimeAsrRealtimeStream {
    fn execution_path(&self) -> RealtimeAsrPath {
        realtime_asr_execution_path(self.variant)
    }

    /// Whether the stream has ended: finished, failed, or expired. A closed
    /// stream rejects every later operation, so callers must stop feeding it.
    pub fn is_closed(&self) -> bool {
        self.resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .closed
    }

    /// The session-wide authorities a rollover hands to the next segment:
    /// the session quota slot, the absolute lifetime and the idle timeout.
    fn segment_authorities(&self) -> Result<(Arc<RealtimeAsrSessionLease>, Instant, Duration)> {
        let resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".into())
        })?;
        Ok((
            session_lease,
            resources.absolute_deadline,
            resources.idle_timeout,
        ))
    }

    fn detach(&self) -> Option<RealtimeAsrDetachedResources> {
        let cleanup = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .detach();
        self.activity.notify_one();
        cleanup
    }

    /// Continues this stream on `next`'s fresh Engine session. The
    /// transcript, chunk numbering and language carry over; the segment
    /// clock restarts. `next` is left holding this stream's closed session.
    fn continue_on(&mut self, next: &mut RuntimeAsrRealtimeStream) {
        std::mem::swap(&mut self.resources, &mut next.resources);
        std::mem::swap(&mut self.activity, &mut next.activity);
        std::mem::swap(&mut self.operation_gate, &mut next.operation_gate);
        self.max_samples = next.max_samples;
        self.committed_samples = 0;
        self.engine_sample_rate = None;
        self.segment.trailing_quiet_samples = 0;
        self.segment.separator_pending = true;
    }

    fn begin_engine_operation(
        &mut self,
        refresh_activity: bool,
    ) -> Result<RealtimeAsrEngineOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let session = resources.engine_session.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("Engine realtime ASR session is unavailable".into())
        })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".into())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".into())
        })?;
        resources.active_operations = resources
            .active_operations
            .checked_add(1)
            .ok_or_else(|| Error::Overloaded("realtime ASR operation count overflowed".into()))?;
        if refresh_activity {
            resources.last_activity = now;
        }
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrEngineOperationHandles {
            session,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    fn map_engine_outputs(
        &mut self,
        outputs: Vec<StreamingOutput>,
    ) -> Vec<RuntimeAsrRealtimeEvent> {
        outputs
            .into_iter()
            .map(|output| {
                let mut delta = output.text.unwrap_or_default();
                if self.segment.separator_pending && !delta.is_empty() {
                    // A new segment's decoder starts without the previous
                    // word, so it may not emit the space between them.
                    self.segment.separator_pending = false;
                    if !self.engine_text.is_empty()
                        && !self.engine_text.ends_with(char::is_whitespace)
                        && !delta.starts_with(char::is_whitespace)
                    {
                        delta.insert(0, ' ');
                    }
                }
                self.engine_text.push_str(&delta);
                let chunk_index = self.engine_chunk_index;
                self.engine_chunk_index = self.engine_chunk_index.saturating_add(1);
                RuntimeAsrRealtimeEvent {
                    delta,
                    text: self.engine_text.clone(),
                    is_final: output.is_final,
                    chunk_index,
                }
            })
            .collect()
    }
    fn begin_operation(&mut self, refresh_activity: bool) -> Result<RealtimeAsrOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let model = resources.model.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream model is unavailable".to_string())
        })?;
        let state = resources.state.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream state is unavailable".to_string())
        })?;
        let execution_contract =
            resources
                .execution_contract
                .as_ref()
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(
                        "realtime ASR stream execution contract is unavailable".to_string(),
                    )
                })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".to_string())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".to_string())
        })?;
        let active_operations = resources.active_operations.checked_add(1).ok_or_else(|| {
            Error::Overloaded("realtime ASR operation count overflowed".to_string())
        })?;
        if refresh_activity {
            resources.last_activity = now;
        }
        resources.active_operations = active_operations;
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrOperationHandles {
            model: model.clone(),
            state,
            execution_contract,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    fn ensure_open(&mut self) -> Result<()> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        Ok(())
    }

    fn close(&self) {
        schedule_realtime_asr_cleanup(self.detach());
    }

    fn close_due_to_timeout(&self) {
        let mut resources = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let reason = resources
            .expiration(Instant::now())
            .unwrap_or("realtime ASR operation timed out");
        let cleanup = resources.detach_expired(reason);
        drop(resources);
        schedule_realtime_asr_cleanup(cleanup);
        self.activity.notify_one();
    }
}

impl Drop for RuntimeAsrRealtimeStream {
    fn drop(&mut self) {
        self.close();
    }
}

fn realtime_asr_session_expiration(
    now: Instant,
    absolute_deadline: Instant,
    idle_deadline: Instant,
    active_operations: usize,
) -> Option<&'static str> {
    if now >= absolute_deadline {
        Some("realtime ASR stream exceeded its absolute lifetime")
    } else if active_operations == 0 && now >= idle_deadline {
        Some("realtime ASR stream exceeded its idle timeout")
    } else {
        None
    }
}

fn spawn_realtime_asr_watchdog(
    resources: &Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
) {
    let weak_resources = Arc::downgrade(resources);
    tokio::spawn(async move {
        loop {
            // Register before reading the deadline so activity between the
            // state snapshot and select cannot be lost.
            let notified = activity.notified();
            let Some(resources) = weak_resources.upgrade() else {
                return;
            };
            let deadline = match resources.lock() {
                Ok(resources) => resources.next_watchdog_deadline(),
                Err(_) => return,
            };
            drop(resources);
            let Some(deadline) = deadline else {
                return;
            };

            tokio::select! {
                _ = tokio::time::sleep_until(deadline.into()) => {
                    let Some(resources) = weak_resources.upgrade() else {
                        return;
                    };
                    let mut resources = match resources.lock() {
                        Ok(resources) => resources,
                        Err(_) => return,
                    };
                    if let Some(reason) = resources.expiration(Instant::now()) {
                        let cleanup = resources.detach_expired(reason);
                        drop(resources);
                        schedule_realtime_asr_cleanup(cleanup);
                        return;
                    }
                }
                _ = notified => {}
            }
        }
    });
}

#[derive(Debug, Clone)]
pub struct RuntimeAsrRealtimeEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

fn map_native_realtime_events(events: Vec<NativeAsrRealtimeEvent>) -> Vec<RuntimeAsrRealtimeEvent> {
    events
        .into_iter()
        .map(|event| RuntimeAsrRealtimeEvent {
            delta: event.delta,
            text: event.text,
            is_final: event.is_final,
            chunk_index: event.chunk_index,
        })
        .collect()
}

fn realtime_state_observation(state: &NativeAsrRealtimeState) -> Result<JobResourceObservation> {
    let (host_bytes, tensor_bytes) = state.resource_usage().ok_or_else(|| {
        Error::InferenceError("realtime ASR state resource usage is unavailable".to_string())
    })?;
    Ok(JobResourceObservation::new(host_bytes, tensor_bytes))
}

fn realtime_state_with_input_observation(
    state: &NativeAsrRealtimeState,
    input_sample_capacity: usize,
) -> Result<JobResourceObservation> {
    let mut observation = realtime_state_observation(state)?;
    let input_bytes = input_sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("realtime ASR input storage overflowed".to_string()))?;
    observation.host_bytes = observation
        .host_bytes
        .checked_add(input_bytes)
        .ok_or_else(|| Error::Overloaded("realtime ASR host usage overflowed".to_string()))?;
    Ok(observation)
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

fn validate_realtime_input_copy(samples: usize, max_samples: usize) -> Result<()> {
    if samples > max_samples {
        return Err(Error::InvalidInput(format!(
            "Realtime ASR input chunk of {samples} samples exceeds the stream reservation of {max_samples} samples"
        )));
    }
    Ok(())
}

async fn run_realtime_blocking_operation<T, B, F>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    execution_contract: LoadedExecutionContract,
    operation_gate: Arc<tokio::sync::Mutex<()>>,
    operation_kind: &'static str,
    build_operation: B,
) -> Result<T>
where
    T: Send + 'static,
    B: FnOnce() -> Result<F> + Send,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let operation_gate = coordinator
        .acquire_job_ordering(job, operation_gate)
        .await?;
    let operation = build_operation()?;
    coordinator
        .run_loaded_blocking_stage(
            job,
            execution_contract,
            WorkUnit::AtomicJob {
                kind: operation_kind.to_string(),
            },
            move || {
                let _operation_gate = operation_gate;
                operation()
            },
        )
        .await
}

fn realtime_stream_resource_vector(
    backend: BackendKind,
    host_bytes: u64,
    tensor_bytes: u64,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Metal => {
            resources.unified_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(tensor_bytes);
        }
    }
    Ok(resources)
}

fn voxtral_realtime_preparation_cost(
    model: &VoxtralModelLease,
    geometry: VoxtralRealtimePreparationGeometry,
) -> Result<WorkCost> {
    let batch = model.realtime_preparation_batch_geometry(&[geometry])?;
    let seal = model.realtime_preparation_stage_seal()?;
    voxtral_realtime_preparation_cost_from_geometry(geometry, batch, seal)
}

fn voxtral_realtime_preparation_cost_from_geometry(
    geometry: VoxtralRealtimePreparationGeometry,
    batch: VoxtralRealtimePreparationBatchGeometry,
    seal: VoxtralRealtimePreparationStageSeal,
) -> Result<WorkCost> {
    if batch.materialized_tensor_elements_per_row > seal.max_materialized_tensor_elements_per_row
        || batch.workspace_per_row_bytes > seal.max_workspace_bytes
    {
        return Err(Error::InferenceError(
            "Voxtral preparation geometry exceeds its loaded stage seal".into(),
        ));
    }
    Ok(WorkCost::new(
        u64::try_from(geometry.source_samples)
            .map_err(|_| Error::Overloaded("Voxtral preparation work overflowed".into()))?
            .max(1),
        batch.materialized_tensor_elements_per_row,
        batch.workspace_per_row_bytes,
    ))
}

fn voxtral_realtime_committed_observation(
    retained_metadata_bytes: u64,
    usage: VoxtralRealtimePreparedResourceUsage,
) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::new(
        retained_metadata_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Overloaded("Voxtral retained metadata overflowed".into()))?
            .checked_add(usage.host_bytes)
            .ok_or_else(|| Error::Overloaded("Voxtral retained host usage overflowed".into()))?,
        usage.tensor_bytes,
    ))
}

fn add_realtime_stream_reservation(
    base: ResourceVector,
    backend: BackendKind,
    reservation: NativeAsrRealtimeResourceReservation,
) -> Result<ResourceVector> {
    base.checked_add(realtime_stream_resource_vector(
        backend,
        reservation.host_bytes(),
        crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES,
    )?)
}

impl RuntimeService {
    async fn start_engine_realtime_stream(
        &self,
        variant: ModelVariant,
        language: Option<&str>,
        prompt: Option<&str>,
        context: RuntimeRequestContext,
        metadata_bytes: usize,
        session_lease: Arc<RealtimeAsrSessionLease>,
        absolute_deadline: Instant,
        idle_timeout: Duration,
    ) -> Result<RuntimeAsrRealtimeStream> {
        if variant.family() == ModelFamily::Voxtral
            && prompt.is_some_and(|prompt| !prompt.trim().is_empty())
        {
            return Err(Error::InvalidInput(
                "Voxtral realtime ASR does not support an initial text prompt".into(),
            ));
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let initial_spec = self.coordinator_job_for_input(
            request_id.clone(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        let initial_job = self
            .coordinator
            .admit_observed(initial_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &initial_job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let (
            model,
            max_samples,
            max_host_bytes,
            max_tensor_bytes,
            max_output_steps,
            max_cache_append,
            voxtral_model,
        ) = match variant.family() {
            ModelFamily::Voxtral => {
                let model = self
                    .model_registry
                    .get_voxtral_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelNotFound(format!("Voxtral model {variant} is not loaded"))
                    })?;
                let peak = model.realtime_stream_peak_reservation()?;
                let max_output_steps = model.realtime_max_output_steps()?;
                let max_cache_append = state_binding
                    .state
                    .managed_kv_runtime()
                    .map(|runtime| runtime.logical_token_reach())
                    .filter(|tokens| *tokens > 0)
                    .and_then(|tokens| usize::try_from(tokens).ok())
                    .ok_or_else(|| {
                        Error::ModelLoadError(
                            "Voxtral realtime state has no finite physical KV-token ceiling".into(),
                        )
                    })?;
                (
                    model.clone(),
                    peak.max_source_samples,
                    peak.max_host_bytes,
                    peak.max_tensor_bytes,
                    max_output_steps,
                    max_cache_append,
                    Some(model),
                )
            }
            _ => {
                return Err(Error::InvalidInput(
                    "unsupported Engine realtime ASR family".into(),
                ))
            }
        };
        let stream_resources = realtime_stream_resource_vector(
            self.backend_context().backend_kind,
            max_host_bytes
                .checked_add(u64::try_from(metadata_bytes).map_err(|_| {
                    Error::Overloaded("realtime ASR retained metadata usage exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    Error::Overloaded(
                        "realtime ASR transactional host reservation overflowed".into(),
                    )
                })?,
            max_tensor_bytes,
        )?;
        let mut execution_spec = initial_job.spec.clone();
        let bridge = self.coordinator.bridge_preparation_admission(initial_job)?;
        execution_spec.resources = execution_spec.resources.checked_add(stream_resources)?;
        let job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution_spec,
                host_input_observation(metadata_bytes)?,
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };

        let mut request = EngineCoreRequest::asr_bytes(Vec::new()).with_model_variant(variant);
        if let Some(language) = language {
            request = request.with_language(language);
        }
        if let Some(prompt) = prompt {
            request = request.with_asr_prompt(prompt);
        }
        request.id = request_id;
        request.workload_class = context.workload_class;
        request.priority = context.priority;
        request.admission_ms = context.admission_ms;
        request.deadline = Some(absolute_deadline);
        request.params.max_tokens = max_cache_append;
        request.bind_execution_adapter(state_binding.execution.clone())?;
        request.bind_v2_state_runtime(
            state_binding.state.clone(),
            state_binding.state.state_fingerprint,
            self.backend_context().backend_kind,
        )?;
        let admission = self.core_engine.start_realtime_asr_session(request);
        let handle = tokio::time::timeout_at(absolute_deadline.into(), admission)
            .await
            .map_err(|_| Error::Timeout("realtime ASR Engine admission".into()))??;
        let engine_session = RuntimeEngineRealtimeSession {
            engine: self.core_engine.clone(),
            handle,
            model,
            retained_metadata_bytes: u64::try_from(metadata_bytes).map_err(|_| {
                Error::Overloaded("realtime ASR retained metadata usage exceeds u64".into())
            })?,
            max_output_steps,
            max_cache_append,
        };
        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: None,
            state: None,
            execution_contract: Some(execution_contract),
            residency_lease: Some(residency_lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: Some(engine_session),
            voxtral_model,
            absolute_deadline,
            idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());
        Ok(RuntimeAsrRealtimeStream {
            variant,
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples,
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
            segment: RealtimeEngineSegment::new(language, prompt),
        })
    }

    pub async fn try_start_asr_realtime_stream(
        &self,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<Option<RuntimeAsrRealtimeStream>> {
        let Some(variant) = resolve_asr_realtime_stream_variant(model_id) else {
            return Ok(None);
        };

        self.observe_broker_capability_request(CapabilityKind::RealtimeAsr, Some(variant), true)?;
        let started_at = Instant::now();
        let limits = self.asr_realtime_sessions.limits;
        let session_lease = self.asr_realtime_sessions.try_acquire()?;
        let absolute_deadline = started_at.checked_add(limits.max_lifetime).ok_or_else(|| {
            Error::ConfigError("realtime ASR absolute lifetime overflowed".to_string())
        })?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Realtime)
            .with_deadline(absolute_deadline);
        let metadata_bytes =
            language.map(str::len).unwrap_or_default() + prompt.map(str::len).unwrap_or_default();
        if realtime_asr_execution_path(variant) == RealtimeAsrPath::Engine {
            return self
                .start_engine_realtime_stream(
                    variant,
                    language,
                    prompt,
                    context,
                    metadata_bytes,
                    session_lease,
                    absolute_deadline,
                    limits.idle_timeout,
                )
                .await
                .map(Some);
        }
        let reservation = NativeAsrModel::conservative_realtime_stream_resource_reservation(
            variant, language, prompt, None,
        )?;
        let mut job_spec = self.coordinator_job_for_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        job_spec.resources = add_realtime_stream_reservation(
            job_spec.resources,
            self.backend_context().backend_kind,
            reservation,
        )?;
        let job = self
            .coordinator
            .admit_observed(job_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let physical_runtime = state_binding
            .state
            .retained_tensor_state_runtime()
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "realtime ASR did not bind its retained tensor runtime".into(),
                )
            })?;
        let model =
            self.model_registry.get_asr(variant).await.ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
        if !model.supports_realtime_stream_decode() {
            return Ok(None);
        }
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not resident")))?;
        let task_model = model.clone();
        let language = language.map(ToOwned::to_owned);
        let prompt = prompt.map(ToOwned::to_owned);
        let retained_metadata_bytes = metadata_bytes
            .checked_add(language.as_ref().map_or(0, String::capacity))
            .and_then(|bytes| bytes.checked_add(prompt.as_ref().map_or(0, String::capacity)))
            .ok_or_else(|| Error::Overloaded("realtime ASR metadata overflowed".to_string()))?;
        job.record_materialized_usage(retained_host_observation(&[
            metadata_bytes,
            language.as_ref().map_or(0, String::capacity),
            prompt.as_ref().map_or(0, String::capacity),
        ])?)?;
        let task_session_lease = session_lease.clone();
        let (state, language, prompt) = self
            .coordinator
            .run_loaded_blocking_stage(
                &job,
                execution_contract.clone(),
                WorkUnit::AtomicJob {
                    kind: "asr.realtime.start".to_string(),
                },
                move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = task_session_lease;
                    let native = task_model.start_realtime_stream_state_with_reservation(
                        language.as_deref(),
                        prompt.as_deref(),
                        None,
                        reservation,
                    )?;
                    let state = RuntimeAsrRealtimeState::new(
                        task_model.as_ref(),
                        native,
                        physical_runtime,
                    )?;
                    Ok((state, language, prompt))
                },
            )
            .await?;
        let steady_usage = realtime_state_observation(&state.native)?;
        job.record_materialized_usage(add_retained_host_bytes(
            steady_usage,
            retained_metadata_bytes,
        )?)?;
        job.prepare_materialized_release(steady_usage)?;
        drop((language, prompt));

        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: Some(model),
            state: Some(Arc::new(StdMutex::new(state))),
            execution_contract: Some(execution_contract),
            residency_lease: Some(lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
            absolute_deadline,
            idle_timeout: limits.idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());

        Ok(Some(RuntimeAsrRealtimeStream {
            variant,
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples: reservation.max_samples(),
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
            segment: RealtimeEngineSegment::default(),
        }))
    }

    pub async fn push_asr_realtime_samples(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        // Reject an impossible single packet before acquiring residency or
        // allocating the owned copy needed by the blocking worker.
        validate_realtime_input_copy(samples.len(), stream.max_samples)?;
        if samples.is_empty() {
            stream.ensure_open()?;
            return Ok(Vec::new());
        }
        if stream.execution_path() == RealtimeAsrPath::Engine {
            let mut events = Vec::new();
            if realtime_segment_rollover_due(
                stream.committed_samples,
                samples.len(),
                stream.segment.trailing_quiet_samples,
                stream.max_samples,
                sample_rate,
            ) {
                // A failed rollover has already finished or detached the
                // current session, so the stream ends instead of rejecting
                // every later push.
                events = match self
                    .roll_over_engine_realtime_segment(stream, sample_rate)
                    .await
                {
                    Ok(events) => events,
                    Err(error) => {
                        stream.close();
                        return Err(error);
                    }
                };
            }
            let total_samples = stream
                .committed_samples
                .checked_add(samples.len())
                .ok_or_else(|| Error::Overloaded("realtime ASR sample clock overflowed".into()))?;
            validate_realtime_input_copy(total_samples, stream.max_samples)?;
            let handles = stream.begin_engine_operation(true)?;
            let operation_lease = self
                .model_lifecycle
                .try_acquire_ready_lease(stream.variant)
                .ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
                });
            let operation_lease = match operation_lease {
                Ok(lease) => lease,
                Err(error) => {
                    stream.close();
                    return Err(error);
                }
            };
            let RealtimeAsrEngineOperationHandles {
                session,
                job,
                session_lease,
                _guard: operation_guard,
            } = handles;
            if stream
                .engine_sample_rate
                .is_some_and(|established| established != sample_rate)
            {
                drop((operation_lease, session_lease, operation_guard));
                return Err(Error::InvalidInput(
                    "realtime ASR sample rate changed within one stream".into(),
                ));
            }
            // Voxtral recomputes its preparation over the committed prefix.
            let geometry = session
                .model
                .realtime_preparation_geometry_for_source_samples(
                    total_samples,
                    sample_rate,
                    VoxtralRealtimePreparationMode::Push,
                )?;
            let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
            let committed_observation = voxtral_realtime_committed_observation(
                session.retained_metadata_bytes,
                session.model.realtime_prepared_resource_usage(geometry)?,
            )?;
            let ordering = match self
                .coordinator
                .acquire_job_ordering(&job, stream.operation_gate.clone())
                .await
            {
                Ok(ordering) => ordering,
                Err(error) => {
                    drop((operation_lease, session_lease, operation_guard));
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            let operation = session
                .engine
                .push_realtime_asr_samples_with_outputs_and_cost(
                    &session.handle,
                    samples.to_vec(),
                    sample_rate,
                    session.max_output_steps,
                    session.max_cache_append,
                    preparation_cost,
                );
            let result = match job.spec.deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout("realtime ASR Engine push".into())),
                },
                None => operation.await,
            };
            drop((ordering, operation_lease, session_lease, operation_guard));
            let (ack, outputs) = match result {
                Ok(result) => result,
                Err(error) => {
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            if ack.accepted_samples() != samples.len() {
                stream.close();
                return Err(Error::InferenceError(
                    "realtime ASR Engine acknowledgement reported the wrong sample span".into(),
                ));
            }
            if let Err(error) = job.record_materialized_usage(committed_observation) {
                stream.close();
                return Err(error);
            }
            stream.committed_samples = total_samples;
            stream.engine_sample_rate = Some(sample_rate);
            stream.segment.trailing_quiet_samples =
                realtime_trailing_quiet_samples(stream.segment.trailing_quiet_samples, samples);
            events.extend(stream.map_engine_outputs(outputs));
            return Ok(events);
        }
        let handles = stream.begin_operation(true)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let observation_job = job.clone();
        let result = run_realtime_blocking_operation(
            &self.coordinator,
            &job,
            execution_contract,
            operation_gate,
            "asr.realtime.push",
            || {
                // The ordering guard is held before this allocation, bounding
                // each stream to one owned packet even after caller timeout.
                let samples = samples.to_vec();
                Ok(move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = session_lease;
                    let _operation_guard = operation_guard;
                    let mut state = state.lock().map_err(|_| {
                        Error::InferenceError("realtime ASR state mutex poisoned".to_string())
                    })?;
                    // Streaming internals replace and release model-owned
                    // tensors while decoding. Restore the full pending claim
                    // before entering that transition so any physical frees
                    // cannot be observed as unclaimed headroom.
                    observation_job
                        .prepare_materialized_release(JobResourceObservation::default())?;
                    let events = state.transact(model.as_ref(), |native| {
                        model.push_realtime_stream_samples(native, &samples, sample_rate)
                    });
                    observation_job.record_materialized_usage(
                        realtime_state_with_input_observation(&state.native, samples.capacity())?,
                    )?;
                    let steady_usage = realtime_state_observation(&state.native)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(samples);
                    events
                })
            },
        )
        .await;
        if let Err(err) = &result {
            if matches!(err, Error::Timeout(_)) {
                stream.close_due_to_timeout();
            } else {
                stream.close();
            }
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    /// Flushes the current Engine session's audio and returns its final
    /// outputs. Success leaves the stream open so the caller decides between
    /// ending it and rolling over into a fresh session.
    async fn finish_engine_realtime_segment(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<StreamingOutput>> {
        let handles = stream.begin_engine_operation(false)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(error) => {
                stream.close();
                return Err(error);
            }
        };
        let RealtimeAsrEngineOperationHandles {
            session,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let sample_rate = stream.engine_sample_rate.ok_or_else(|| {
            Error::InvalidInput("realtime ASR finish requires committed audio".into())
        })?;
        let geometry = session
            .model
            .realtime_preparation_geometry_for_source_samples(
                stream.committed_samples,
                sample_rate,
                VoxtralRealtimePreparationMode::Finish,
            )?;
        let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
        let committed_observation = voxtral_realtime_committed_observation(
            session.retained_metadata_bytes,
            session.model.realtime_prepared_resource_usage(geometry)?,
        )?;
        let ordering = match self
            .coordinator
            .acquire_job_ordering(&job, stream.operation_gate.clone())
            .await
        {
            Ok(ordering) => ordering,
            Err(error) => {
                drop((operation_lease, session_lease, operation_guard));
                if matches!(error, Error::Timeout(_)) {
                    stream.close_due_to_timeout();
                } else {
                    stream.close();
                }
                return Err(error);
            }
        };
        let operation = session
            .engine
            .finish_realtime_asr_session_with_outputs_and_cost(
                &session.handle,
                session.max_output_steps,
                session.max_cache_append,
                preparation_cost,
            );
        let result = match job.spec.deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                Ok(result) => result,
                Err(_) => Err(Error::Timeout("realtime ASR Engine finish".into())),
            },
            None => operation.await,
        };
        drop((ordering, operation_lease, session_lease, operation_guard));
        let (_ack, outputs) = match result {
            Ok(result) => result,
            Err(error) => {
                if matches!(error, Error::Timeout(_)) {
                    stream.close_due_to_timeout();
                } else {
                    stream.close();
                }
                return Err(error);
            }
        };
        if let Err(error) = job.record_materialized_usage(committed_observation) {
            stream.close();
            return Err(error);
        }
        Ok(outputs)
    }

    /// Ends the current Engine session at a segment boundary and continues
    /// the stream in a fresh one. The flushed text is returned as ordinary
    /// (non-final) events; the stream's transcript stays continuous.
    async fn roll_over_engine_realtime_segment(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        if stream
            .engine_sample_rate
            .is_some_and(|established| established != sample_rate)
        {
            return Err(Error::InvalidInput(
                "realtime ASR sample rate changed within one stream".into(),
            ));
        }
        let (session_lease, absolute_deadline, idle_timeout) = stream.segment_authorities()?;
        let outputs = self.finish_engine_realtime_segment(stream).await?;
        let mut events = stream.map_engine_outputs(outputs);
        for event in &mut events {
            event.is_final = false;
        }
        // The session quota slot travels with `session_lease`; the finished
        // session's memory and residency are released before the next one
        // is admitted.
        tokio::time::timeout_at(
            absolute_deadline.into(),
            release_realtime_asr_cleanup(stream.detach()),
        )
        .await
        .map_err(|_| Error::Timeout("realtime ASR segment cleanup".into()))??;
        let language = stream.segment.language.clone();
        let prompt = stream.segment.prompt.clone();
        let metadata_bytes = language.as_deref().map(str::len).unwrap_or_default()
            + prompt.as_deref().map(str::len).unwrap_or_default();
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Realtime)
            .with_deadline(absolute_deadline);
        let mut next = self
            .start_engine_realtime_stream(
                stream.variant,
                language.as_deref(),
                prompt.as_deref(),
                context,
                metadata_bytes,
                session_lease,
                absolute_deadline,
                idle_timeout,
            )
            .await?;
        stream.continue_on(&mut next);
        Ok(events)
    }

    pub async fn finish_asr_realtime_stream(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        if stream.execution_path() == RealtimeAsrPath::Engine {
            let outputs = self.finish_engine_realtime_segment(stream).await?;
            stream.close();
            return Ok(stream.map_engine_outputs(outputs));
        }
        let handles = stream.begin_operation(false)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let observation_job = job.clone();
        let result = run_realtime_blocking_operation(
            &self.coordinator,
            &job,
            execution_contract,
            operation_gate,
            "asr.realtime.finish",
            || {
                Ok(move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = session_lease;
                    let _operation_guard = operation_guard;
                    let mut state = state.lock().map_err(|_| {
                        Error::InferenceError("realtime ASR state mutex poisoned".to_string())
                    })?;
                    observation_job
                        .prepare_materialized_release(JobResourceObservation::default())?;
                    state.transact(model.as_ref(), |native| {
                        model.finish_realtime_stream(native)
                    })
                })
            },
        )
        .await;
        if matches!(&result, Err(Error::Timeout(_))) {
            stream.close_due_to_timeout();
        } else {
            stream.close();
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    pub fn asr_realtime_stream_variant(&self, stream: &RuntimeAsrRealtimeStream) -> ModelVariant {
        stream.variant
    }

    async fn asr_transcribe_audio_chat<F>(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        max_tokens: Option<usize>,
        streaming_required: bool,
        on_delta: F,
        request_id: String,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let input_bytes = audio_input.input_bytes();
        if input_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let spec = self.coordinator_job_for_audio_input(
            request_id,
            CoordinatorLane::Atomic,
            runtime_context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;

        // Establish an owned packet only after admission. The caller-owned
        // allocation remains live while this future runs, so both allocations
        // are reconciled until the blocking decoder consumes the owned copy.
        let audio_input = audio_input.into_owned_for_job(&job).await?;
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            audio_input.retained_bytes(),
        ])?)?;

        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::Asr,
                streaming_required,
                ExecutionTargetKind::DirectModel,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })?;
        let observation_job = job.clone();
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_workspace(
                &job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: "asr.audio_chat".to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let retained_audio_bytes = audio_input.retained_bytes();
                    let (samples, sample_rate) = audio_input.decode()?;
                    let steady_usage = decoded_audio_observation(input_bytes, samples.capacity())?;
                    observation_job.record_materialized_usage(add_retained_host_bytes(
                        steady_usage,
                        retained_audio_bytes,
                    )?)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(audio_input);
                    let duration_secs = if sample_rate > 0 {
                        samples.len() as f32 / sample_rate as f32
                    } else {
                        0.0
                    };
                    let mut on_delta = on_delta;
                    let mut delta_sink = |delta: &str| {
                        if !delta.is_empty() {
                            on_delta(delta.to_string());
                        }
                    };
                    let output = model
                        .transcribe_with_callback_and_max_tokens_from_invocation_workspace(
                            &samples,
                            sample_rate,
                            max_tokens,
                            leases,
                            &mut delta_sink,
                        )?;

                    Ok(AsrTranscription {
                        text: asr_transcript_text(&output.text),
                        language: output.language,
                        duration_secs,
                        asr_diagnostics: output.diagnostics,
                    })
                },
            )
            .await
    }

    async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
        native_word_timestamps: bool,
    ) -> Result<AdmittedEngineRequest> {
        validate_asr_language(variant, language)?;
        let audio_bytes = match audio_input {
            AsrAudioInput::Base64("") => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio input".to_string(),
                ));
            }
            AsrAudioInput::Bytes([]) => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio bytes".to_string(),
                ));
            }
            AsrAudioInput::Base64(audio) => audio.len(),
            AsrAudioInput::Bytes(audio) => audio.len(),
        };
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .and_then(|bytes| bytes.checked_add(prompt.map(str::len).unwrap_or_default()))
            .and_then(|bytes| bytes.checked_add(correlation_id.map(str::len).unwrap_or_default()))
            .ok_or_else(|| Error::Overloaded("ASR preparation input overflowed".to_string()))?;
        self.prepare_engine_request_blocking_with_input(
            variant,
            TaskType::ASR,
            streaming,
            runtime_context,
            input_bytes,
            ResourceVector::zero(),
            move |job| async move {
                let audio_input = audio_input.into_owned_for_job(&job).await?;
                let language =
                    copy_optional_preparation_string(&job, language, "ASR language metadata")
                        .await?;
                let prompt =
                    copy_optional_preparation_string(&job, prompt, "ASR prompt metadata").await?;
                let correlation_id = copy_optional_preparation_string(
                    &job,
                    correlation_id,
                    "ASR correlation metadata",
                )
                .await?;
                let retained_metadata_bytes = language
                    .as_ref()
                    .map_or(0, String::capacity)
                    .checked_add(prompt.as_ref().map_or(0, String::capacity))
                    .and_then(|bytes| {
                        bytes.checked_add(correlation_id.as_ref().map_or(0, String::capacity))
                    })
                    .ok_or_else(|| {
                        Error::Overloaded("ASR retained metadata overflowed".to_string())
                    })?;
                job.record_materialized_usage(retained_host_observation(&[
                    input_bytes,
                    audio_input.retained_bytes(),
                    retained_metadata_bytes,
                ])?)?;
                Ok((audio_input, language, prompt, correlation_id))
            },
            move |_registry, (audio_input, language, prompt, correlation_id)| {
                audio_input.validate_source_size()?;
                let runtime_request = match audio_input {
                    OwnedAsrAudioInput::Base64(audio_base64) => AsrRuntimeRequest::from_base64(
                        audio_base64,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                    OwnedAsrAudioInput::Bytes(audio_bytes) => AsrRuntimeRequest::from_bytes(
                        audio_bytes,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                }
                .with_prompt(prompt)
                .with_native_word_timestamps(native_word_timestamps);
                let mut request = runtime_request.into_engine_request();
                apply_asr_max_tokens(&mut request, variant, max_tokens);
                Ok(request)
            },
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            let context = RuntimeRequestContext::default();
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                false,
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(&output.text.unwrap_or_default());

        Ok(AsrTranscription {
            text,
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), true)?;
            let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Streaming);
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    true,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                true,
                false,
            )
            .await?;
        self.stream_admitted_asr_request(admitted, variant, true, language, on_delta, |_| {})
            .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
            false,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
                runtime_context,
                false,
                native_word_timestamps,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(&output.text.unwrap_or_default());

        Ok(AsrTranscription {
            text,
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress<
        F,
        P,
    >(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            on_progress,
            true,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress<F, P>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
        broker_streaming_required: bool,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(
                CapabilityKind::Asr,
                Some(variant),
                broker_streaming_required,
            )?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    broker_streaming_required,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
                runtime_context,
                true,
                false,
            )
            .await?;
        self.stream_admitted_asr_request(
            admitted,
            variant,
            broker_streaming_required,
            language,
            on_delta,
            on_progress,
        )
        .await
    }

    /// Run an admitted ASR request as a stream of text deltas.
    ///
    /// `native_stream_requested` asks for the model's native streaming
    /// contract. A model whose ASR adapter declares no streaming mode is
    /// offline-only, so its stream runs on the offline route instead: the
    /// transport still carries the deltas its decode emits (retained per-step
    /// text, chunk partials). Either way the deltas end with whatever part of
    /// the final transcript they did not cover, so they concatenate to it.
    async fn stream_admitted_asr_request<F, P>(
        &self,
        admitted: AdmittedEngineRequest,
        variant: ModelVariant,
        native_stream_requested: bool,
        language: Option<&str>,
        mut on_delta: F,
        mut on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let mut streamed_text = String::new();
        let handle_chunk = |chunk: crate::engine::StreamingOutput| {
            if let Some(progress) = chunk.asr_progress {
                on_progress(progress);
            }
            if let Some(delta) = chunk.text {
                if !delta.is_empty() {
                    streamed_text.push_str(&delta);
                    on_delta(delta);
                }
            }
            std::future::ready(Ok(()))
        };
        let output =
            if native_stream_requested && asr_streams_natively(&self.adapter_registry, variant) {
                self.run_admitted_streaming_request(admitted, handle_chunk)
                    .await?
            } else {
                self.run_admitted_transport_streaming_request(admitted, handle_chunk)
                    .await?
            };
        let duration_secs = asr_transcription_duration_secs(&output);
        let text = asr_transcript_text(output.text.as_deref().unwrap_or(&streamed_text));
        if let Some(tail) = unstreamed_transcript_tail(&streamed_text, &text) {
            on_delta(tail.to_string());
        }

        Ok(AsrTranscription {
            text,
            language: asr_response_language(language, output.asr_diagnostics.as_ref()),
            duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    /// Transcribe audio with Voxtral through the offline transcription path.
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
        )
        .await
    }

    /// Transcribe audio with Voxtral and emit incremental deltas from offline decode.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_correlation(audio_base64, model_id, language, None)
            .await
    }

    /// Transcribe audio with request correlation metadata.
    pub async fn asr_transcribe_with_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata.
    pub async fn asr_transcribe_with_prompt_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR prompt and max-token decode budget.
    pub async fn asr_transcribe_with_prompt_max_tokens_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    pub async fn asr_transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_prompt(audio_bytes, model_id, language, None)
            .await
    }

    pub async fn asr_transcribe_bytes_with_prompt(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_prompt_max_tokens_and_correlation(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
            audio_bytes,
            model_id,
            language,
            prompt,
            max_tokens,
            correlation_id,
            runtime_context,
            false,
        )
        .await
    }

    /// Like [`Self::asr_transcribe_bytes_with_runtime_context`], optionally
    /// asking for model-native word timestamps: a model that emits them
    /// (Granite Speech Plus) returns them as `timestamp_words` in its
    /// diagnostics; any other model transcribes as usual.
    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context_and_word_timestamps(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        native_word_timestamps: bool,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            runtime_context,
            native_word_timestamps,
        )
        .await
    }

    /// Transcribe audio and emit deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_correlation(
            audio_base64,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio and emit deltas with request correlation metadata.
    pub async fn asr_transcribe_streaming_with_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata and deltas.
    pub async fn asr_transcribe_streaming_with_prompt_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_prompt_and_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_runtime_context<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
            true,
            runtime_context,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            true,
            runtime_context,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            runtime_context,
        )
        .await
    }

    pub async fn speaker_attributed_asr(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Base64(audio_base64),
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_bytes_with_progress(
            audio_bytes,
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes_with_progress<P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Bytes(audio_bytes),
            model_id,
            language,
            min_speakers,
            max_speakers,
            on_progress,
        )
        .await
    }

    async fn speaker_attributed_asr_input_with_progress<P>(
        &self,
        audio_input: AsrAudioInput<'_>,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        mut on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_speaker_attributed_asr_variant(model_id)?;
        validate_asr_language(variant, language)?;
        let audio_bytes = audio_input.input_bytes();
        if audio_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .ok_or_else(|| {
                Error::Overloaded("speaker-attributed ASR input overflowed".to_string())
            })?;
        self.observe_broker_capability_request(
            CapabilityKind::SpeakerAttributedAsr,
            Some(variant),
            false,
        )?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Background);
        let spec = self.coordinator_job_for_audio_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Pipeline,
            context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;
        let owned_audio = audio_input.into_owned_for_job(&job).await?;
        let language_owned =
            copy_optional_preparation_string(&job, language, "speaker-attributed ASR language")
                .await?;
        let language_bytes = language_owned.as_ref().map_or(0, String::capacity);
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            owned_audio.retained_bytes(),
            language_bytes,
        ])?)?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::SpeakerAttributedAsr,
                false,
                ExecutionTargetKind::PipelineRunner,
                CapabilityLoadOptions::default(),
            )
            .await?;
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let observation_job = job.clone();
        let (residency_lease, samples, sample_rate) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let retained_audio_bytes = owned_audio.retained_bytes();
                let (samples, sample_rate) = owned_audio.decode()?;
                let samples = Arc::<[f32]>::from(samples);
                let steady_usage = decoded_audio_observation(
                    input_bytes.checked_add(language_bytes).ok_or_else(|| {
                        Error::Overloaded(
                            "speaker-attributed ASR retained input overflowed".to_string(),
                        )
                    })?,
                    samples.len(),
                )?;
                observation_job.record_materialized_usage(add_retained_host_bytes(
                    steady_usage,
                    retained_audio_bytes,
                )?)?;
                observation_job.prepare_materialized_release(steady_usage)?;
                drop(owned_audio);
                Ok((residency_lease, samples, sample_rate))
            })
            .await?;
        let duration_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let retained_input_bytes = input_bytes.checked_add(language_bytes).ok_or_else(|| {
            Error::Overloaded("speaker-attributed ASR retained input overflowed".to_string())
        })?;
        // FullTranscript prefix mode grows every chunk's prompt with the
        // prior transcript; reserve headroom for the capped prefix so chunk
        // sizing leaves room for it instead of failing mid-job.
        let prefix_token_allowance = match GraniteSaaPrefixMode::from_env() {
            GraniteSaaPrefixMode::None => 0usize,
            GraniteSaaPrefixMode::FullTranscript => GRANITE_SAA_PREFIX_MAX_CHARS / 4,
        };
        let model_limit_secs = compose_invocation_audio_limit(
            model.max_audio_seconds_hint(),
            model.audio_token_rate(),
            self.model_registry.effective_context(variant),
            move |secs| {
                granite_saa_max_new_tokens_for_duration(secs)
                    .saturating_add(prefix_token_allowance)
            },
        )?;
        tracing::info!(
            variant = %variant,
            model_limit_secs = ?model_limit_secs,
            "resolved speaker-attributed ASR invocation audio limit"
        );
        if granite_saa_should_use_single_pass(duration_secs, model_limit_secs) {
            let task_language = language_owned.clone();
            let task_model = model.clone();
            let max_new_tokens = granite_saa_max_new_tokens(&samples, sample_rate);
            let observation_job = job.clone();
            let transcription =
                self.coordinator
                    .run_loaded_blocking_stage_with_invocation_paged(
                        &job,
                        execution_contract,
                        state_binding,
                        WorkUnit::PipelineStage {
                            name: "asr.speaker_attributed.decode".to_string(),
                            ordinal: 0,
                        },
                        move |leases| {
                            let _residency_lease = residency_lease;
                            observation_job.record_materialized_usage(
                                decoded_audio_observation(retained_input_bytes, samples.len())?,
                            )?;
                            granite_saa_transcribe_chunk(
                                &task_model,
                                &samples,
                                sample_rate,
                                task_language.as_deref(),
                                None,
                                max_new_tokens,
                                granite_saa_single_invocation_cache(leases)?,
                            )
                        },
                    )
                    .await?;

            return Ok(speaker_attributed_asr_result_from_text_with_warnings(
                transcription.text.as_str(),
                transcription.language.or(language_owned),
                duration_secs,
                min_speakers,
                max_speakers,
                Vec::new(),
            ));
        }

        let long_form = granite_saa_long_form_transcribe(
            &self.coordinator,
            &job,
            model,
            execution_contract,
            state_binding,
            samples,
            sample_rate,
            language_owned.as_deref(),
            model_limit_secs,
            residency_lease,
            retained_input_bytes,
            &mut on_progress,
        )
        .await?;

        Ok(speaker_attributed_asr_result_from_text_with_warnings(
            long_form.text.as_str(),
            long_form.language.or(language_owned),
            duration_secs,
            min_speakers,
            max_speakers,
            long_form.warnings,
        ))
    }

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
            reference_text,
            language,
            model_id,
        )
        .await
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
            reference_text,
            language,
            model_id,
        )
        .await
    }

    async fn force_align_input(
        &self,
        audio_input: AsrAudioInput<'_>,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = resolve_forced_aligner_variant(model_id)?;
        if audio_input.input_bytes() == 0 {
            return Err(Error::InvalidInput(
                "forced alignment request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        if reference_text.trim().is_empty() {
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
            .checked_add(reference_text.len())
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
        let reference_text =
            copy_preparation_string(&job, reference_text, "forced alignment transcript").await?;
        let language =
            copy_optional_preparation_string(&job, language, "forced alignment language").await?;
        let owned_metadata_bytes = reference_text
            .capacity()
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
        self.coordinator
            .run_loaded_blocking_stage(
                &job,
                execution_contract,
                WorkUnit::AtomicJob {
                    kind: "asr.forced_alignment".to_string(),
                },
                move || {
                    let _residency_lease = residency_lease;
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
                    let steady_usage =
                        decoded_audio_observation(retained_bytes, samples.capacity())?;
                    observation_job.record_materialized_usage(add_retained_host_bytes(
                        steady_usage,
                        retained_audio_bytes,
                    )?)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(audio_input);
                    model.force_align(&samples, sample_rate, &reference_text, language.as_deref())
                },
            )
            .await
    }
}

fn resolve_speaker_attributed_asr_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::GraniteSpeech412BPlus,
    };

    if variant != ModelVariant::GraniteSpeech412BPlus {
        return Err(Error::InvalidInput(format!(
            "Speaker attributed ASR currently requires Granite-Speech-4.1-2B-Plus, got {variant}"
        )));
    }

    Ok(variant)
}

#[derive(Debug, Clone)]
struct GraniteSaaLongFormOutput {
    text: String,
    language: Option<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct GraniteSaaChunkDecodeDiagnostics {
    generated_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    stop_reason: Option<String>,
    prompt_tokens: Option<usize>,
    prompt_prefix_tokens: Option<usize>,
}

#[derive(Debug, Default)]
struct GraniteSaaTranscriptAssembler {
    turns: Vec<SpeakerAttributedAsrTurn>,
}

impl GraniteSaaTranscriptAssembler {
    fn push_chunk_text(&mut self, chunk_text: &str, chunk_index: usize) -> Vec<String> {
        let mut warnings = Vec::new();
        let incoming = speaker_attributed_asr_turns_from_text(chunk_text);
        if incoming.is_empty() {
            if !chunk_text.trim().is_empty() {
                warnings.push(format!(
                    "Granite SAA chunk {} returned text without speaker labels.",
                    chunk_index + 1
                ));
            }
            return warnings;
        }

        let mut speaker_aliases = HashMap::<String, String>::new();
        if let (Some(last), Some(first)) = (self.turns.last(), incoming.first()) {
            let overlap = overlap_prefix_word_count(
                last.text.as_str(),
                first.text.as_str(),
                GRANITE_SAA_MIN_OVERLAP_WORDS,
                GRANITE_SAA_MAX_OVERLAP_WORDS,
            );
            if overlap > 0
                && first.speaker != UNKNOWN_SAA_SPEAKER
                && last.speaker != UNKNOWN_SAA_SPEAKER
                && first.speaker != last.speaker
            {
                speaker_aliases.insert(first.speaker.clone(), last.speaker.clone());
                warnings.push(format!(
                    "Granite SAA chunk {} reused speaker label {} across an overlap; mapped it to {}.",
                    chunk_index + 1,
                    first.speaker,
                    last.speaker
                ));
            }
        }
        if !speaker_aliases.is_empty() {
            let mut known_global_speakers = Vec::<String>::new();
            for turn in &self.turns {
                if turn.speaker != UNKNOWN_SAA_SPEAKER
                    && !known_global_speakers.contains(&turn.speaker)
                {
                    known_global_speakers.push(turn.speaker.clone());
                }
            }

            let mut used_global_speakers = speaker_aliases
                .values()
                .cloned()
                .collect::<HashSet<String>>();
            let mut local_speakers = Vec::<String>::new();
            for turn in &incoming {
                if turn.speaker != UNKNOWN_SAA_SPEAKER && !local_speakers.contains(&turn.speaker) {
                    local_speakers.push(turn.speaker.clone());
                }
            }

            for local_speaker in local_speakers {
                if speaker_aliases.contains_key(local_speaker.as_str())
                    || !used_global_speakers.contains(local_speaker.as_str())
                {
                    continue;
                }
                if let Some(global_speaker) = known_global_speakers
                    .iter()
                    .find(|speaker| !used_global_speakers.contains(speaker.as_str()))
                    .cloned()
                {
                    speaker_aliases.insert(local_speaker.clone(), global_speaker.clone());
                    used_global_speakers.insert(global_speaker.clone());
                    warnings.push(format!(
                        "Granite SAA chunk {} remapped local speaker label {} to {} after detecting a label reset.",
                        chunk_index + 1,
                        local_speaker,
                        global_speaker
                    ));
                }
            }
        }

        for mut turn in incoming {
            if let Some(mapped) = speaker_aliases.get(turn.speaker.as_str()) {
                turn.speaker = mapped.clone();
            }
            self.push_turn(turn);
        }

        warnings
    }

    fn push_turn(&mut self, mut turn: SpeakerAttributedAsrTurn) {
        turn.text = turn.text.trim().to_string();
        if turn.text.is_empty() {
            return;
        }

        let Some(last) = self.turns.last_mut() else {
            self.turns.push(turn);
            return;
        };

        let overlap = overlap_prefix_word_count(
            last.text.as_str(),
            turn.text.as_str(),
            GRANITE_SAA_MIN_OVERLAP_WORDS,
            GRANITE_SAA_MAX_OVERLAP_WORDS,
        );
        if overlap > 0 {
            turn.text = drop_prefix_words(turn.text.as_str(), overlap);
            if turn.text.trim().is_empty() {
                return;
            }
        }

        if last.speaker == turn.speaker {
            append_with_spacing(&mut last.text, turn.text.trim());
        } else {
            self.turns.push(turn);
        }
    }

    fn text(&self) -> String {
        format_saa_turns(&self.turns)
    }

    fn prefix_text(&self) -> String {
        let mut selected = Vec::<String>::new();
        let mut selected_chars = 0usize;

        for turn in self.turns.iter().rev().take(GRANITE_SAA_PREFIX_MAX_TURNS) {
            let formatted = format_saa_turn(turn);
            let separator_chars = usize::from(!selected.is_empty());
            let formatted_chars = formatted.chars().count();
            if selected_chars + separator_chars + formatted_chars <= GRANITE_SAA_PREFIX_MAX_CHARS {
                selected.push(formatted);
                selected_chars += separator_chars + formatted_chars;
                continue;
            }

            if selected.is_empty() {
                let label_overhead = turn.speaker.chars().count() + "[]: ".chars().count();
                let text_budget = GRANITE_SAA_PREFIX_MAX_CHARS.saturating_sub(label_overhead);
                let suffix = recent_word_suffix(turn.text.as_str(), text_budget);
                if !suffix.is_empty() {
                    selected.push(format!("[{}]: {}", turn.speaker, suffix));
                }
            }
            break;
        }

        selected.reverse();
        selected.join(" ")
    }
}

fn speaker_attributed_asr_turns_from_text(raw_text: &str) -> Vec<SpeakerAttributedAsrTurn> {
    let parsed = parse_granite_speech_output(raw_text);
    if parsed.segments.is_empty() {
        let text = parsed.text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        return vec![SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: text.to_string(),
            start_secs: None,
            end_secs: None,
        }];
    }

    parsed
        .segments
        .into_iter()
        .map(|segment| SpeakerAttributedAsrTurn {
            speaker: segment
                .speaker
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(UNKNOWN_SAA_SPEAKER)
                .to_string(),
            text: segment.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        })
        .filter(|turn| !turn.text.is_empty())
        .fold(Vec::<SpeakerAttributedAsrTurn>::new(), |mut turns, turn| {
            if let Some(last) = turns.last_mut() {
                if last.speaker == turn.speaker {
                    append_with_spacing(&mut last.text, turn.text.as_str());
                    return turns;
                }
            }
            turns.push(turn);
            turns
        })
}

fn granite_saa_should_use_single_pass(duration_secs: f32, model_limit_secs: Option<f32>) -> bool {
    let limit = model_limit_secs
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    duration_secs.is_finite() && duration_secs <= limit
}

fn granite_saa_long_form_config(model_limit_secs: Option<f32>) -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig {
        target_chunk_secs: GRANITE_SAA_TARGET_CHUNK_SECS,
        hard_max_chunk_secs: GRANITE_SAA_HARD_MAX_CHUNK_SECS,
        overlap_secs: GRANITE_SAA_OVERLAP_SECS,
        min_chunk_secs: GRANITE_SAA_MIN_CHUNK_SECS,
        silence_search_secs: GRANITE_SAA_SILENCE_SEARCH_SECS,
        min_word_overlap: GRANITE_SAA_MIN_OVERLAP_WORDS,
        max_word_overlap: GRANITE_SAA_MAX_OVERLAP_WORDS,
        min_context_replay_words: GRANITE_SAA_MIN_OVERLAP_WORDS,
        max_context_replay_words: GRANITE_SAA_MAX_OVERLAP_WORDS,
        ..Default::default()
    };

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    let overrides = GraniteSaaChunkOverrides::get();
    if let Some(value) = overrides.target_chunk_secs {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = overrides.hard_max_chunk_secs {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = overrides.overlap_secs {
        cfg.overlap_secs = value;
    }
    if let Some(value) = overrides.min_chunk_secs {
        cfg.min_chunk_secs = value;
    }
    if let Some(value) = overrides.silence_search_secs {
        cfg.silence_search_secs = value;
    }

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }

    // A context-derived limit below the family's minimum chunk size forces
    // graceful degradation: smaller chunks keep Speaker-Attributed ASR
    // functional on constrained hosts instead of failing every invocation.
    if let Some(limit) = model_limit_secs
        .filter(|value| value.is_finite() && *value > 0.0 && *value < cfg.min_chunk_secs)
    {
        tracing::warn!(
            context_limit_secs = limit,
            family_min_chunk_secs = cfg.min_chunk_secs,
            "Granite SAA invocation context limits chunks below the family minimum; speaker continuity degrades"
        );
        cfg.min_chunk_secs = limit.max(1.0);
    }

    cfg.hard_max_chunk_secs = cfg
        .hard_max_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    cfg.target_chunk_secs = cfg
        .target_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(cfg.hard_max_chunk_secs);
    cfg.overlap_secs = cfg.overlap_secs.clamp(0.0, cfg.target_chunk_secs * 0.45);
    cfg
}

/// The `IZWI_GRANITE_SAA_CHUNK_*` overrides, read once per process.
#[derive(Debug, Clone, Copy)]
struct GraniteSaaChunkOverrides {
    target_chunk_secs: Option<f32>,
    hard_max_chunk_secs: Option<f32>,
    overlap_secs: Option<f32>,
    min_chunk_secs: Option<f32>,
    silence_search_secs: Option<f32>,
}

impl GraniteSaaChunkOverrides {
    fn get() -> Self {
        static OVERRIDES: crate::env::ReadOnce<GraniteSaaChunkOverrides> =
            crate::env::ReadOnce::new(|| GraniteSaaChunkOverrides {
                target_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_TARGET_SECS"),
                hard_max_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MAX_SECS"),
                overlap_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_OVERLAP_SECS"),
                min_chunk_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MIN_SECS"),
                silence_search_secs: env_positive_f32("IZWI_GRANITE_SAA_CHUNK_SILENCE_SEARCH_SECS"),
            });
        OVERRIDES.get()
    }
}

fn granite_saa_chunk_plan(
    samples: &[f32],
    sample_rate: u32,
    model_limit_secs: Option<f32>,
) -> Vec<AudioChunk> {
    let cfg = granite_saa_long_form_config(model_limit_secs);
    plan_audio_chunks(samples, sample_rate, &cfg, Some(cfg.hard_max_chunk_secs))
}

fn granite_saa_max_new_tokens_for_duration(duration_secs: f32) -> usize {
    static OVERRIDE: crate::env::ReadOnce<Option<usize>> =
        crate::env::ReadOnce::new(|| env_positive_usize("IZWI_GRANITE_SAA_MAX_NEW_TOKENS"));
    if let Some(override_tokens) = OVERRIDE.get() {
        return override_tokens;
    }

    let duration_budget = if duration_secs.is_finite() && duration_secs > 0.0 {
        (duration_secs * GRANITE_SAA_NEW_TOKENS_PER_SECOND).ceil() as usize
    } else {
        0
    };
    GRANITE_SAA_NEW_TOKEN_RESERVE
        .saturating_add(duration_budget)
        .clamp(GRANITE_SAA_MIN_NEW_TOKENS, GRANITE_SAA_MAX_NEW_TOKENS)
}

fn granite_saa_max_new_tokens(audio: &[f32], sample_rate: u32) -> usize {
    let duration_secs = if sample_rate > 0 {
        audio.len() as f32 / sample_rate as f32
    } else {
        0.0
    };
    granite_saa_max_new_tokens_for_duration(duration_secs)
}

fn granite_saa_chunk_prefix_text(
    assembler: &GraniteSaaTranscriptAssembler,
    mode: GraniteSaaPrefixMode,
) -> Option<String> {
    match mode {
        GraniteSaaPrefixMode::None => None,
        GraniteSaaPrefixMode::FullTranscript => {
            let prefix_text = assembler.prefix_text();
            (!prefix_text.trim().is_empty()).then_some(prefix_text)
        }
    }
}

fn granite_saa_decode_diagnostics(
    diagnostics: Option<&serde_json::Value>,
) -> GraniteSaaChunkDecodeDiagnostics {
    let Some(diagnostics) = diagnostics else {
        return GraniteSaaChunkDecodeDiagnostics::default();
    };
    let decode = diagnostics.get("decode").unwrap_or(diagnostics);
    let prompt = diagnostics.get("prompt");

    GraniteSaaChunkDecodeDiagnostics {
        generated_tokens: decode
            .get("generated_tokens")
            .or_else(|| diagnostics.get("generated_tokens"))
            .and_then(json_usize),
        max_new_tokens: decode.get("max_new_tokens").and_then(json_usize),
        stop_reason: decode
            .get("stop_reason")
            .or_else(|| diagnostics.get("stop_reason"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        prompt_tokens: prompt
            .and_then(|value| value.get("prompt_tokens"))
            .or_else(|| diagnostics.get("prompt_tokens"))
            .and_then(json_usize),
        prompt_prefix_tokens: prompt
            .and_then(|value| value.get("prefix_tokens"))
            .or_else(|| diagnostics.get("prompt_prefix_tokens"))
            .and_then(json_usize),
    }
}

fn granite_saa_max_token_warning(
    chunk_index: usize,
    diagnostics: &GraniteSaaChunkDecodeDiagnostics,
) -> Option<String> {
    if diagnostics.stop_reason.as_deref() != Some("max_tokens") {
        return None;
    }

    let generated = diagnostics
        .generated_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let max_new_tokens = diagnostics
        .max_new_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "configured".to_string());
    Some(format!(
        "Granite SAA chunk {} reached the decode token limit ({generated}/{max_new_tokens}); transcript may be incomplete.",
        chunk_index + 1
    ))
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

async fn granite_saa_long_form_transcribe<P>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    model: Arc<NativeAsrModel>,
    execution_contract: LoadedExecutionContract,
    state_binding: LoadedCapabilityBinding,
    samples: Arc<[f32]>,
    sample_rate: u32,
    language: Option<&str>,
    model_limit_secs: Option<f32>,
    residency_lease: ModelResidencyLease,
    retained_input_bytes: usize,
    on_progress: &mut P,
) -> Result<GraniteSaaLongFormOutput>
where
    P: FnMut(AsrProgress) + Send + 'static,
{
    let planning_samples = samples.clone();
    let planning_job = job.clone();
    let (mut residency_lease, chunks) = coordinator
        .run_host_blocking_stage(job, move || {
            planning_job.record_materialized_usage(decoded_audio_observation(
                retained_input_bytes,
                planning_samples.len(),
            )?)?;
            Ok((
                residency_lease,
                granite_saa_chunk_plan(&planning_samples, sample_rate, model_limit_secs),
            ))
        })
        .await?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Granite SAA chunk planner produced no chunks".to_string(),
        ));
    }

    on_progress(granite_saa_processing_progress(&chunks, sample_rate));

    let mut assembler = GraniteSaaTranscriptAssembler::default();
    let mut language_out = language.map(ToOwned::to_owned);
    let mut warnings = vec![format!(
        "Granite SAA processed long audio in {} chunks; speaker label continuity across chunks is best-effort.",
        chunks.len()
    )];

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
            warnings.push(format!(
                "Granite SAA skipped invalid chunk {} with sample range {}..{}.",
                idx + 1,
                chunk.start_sample,
                chunk.end_sample
            ));
            continue;
        }

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkStarted,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));

        let chunk_start = chunk.start_sample;
        let chunk_end = chunk.end_sample;
        let task_model = model.clone();
        let task_samples = samples.clone();
        let observation_job = job.clone();
        let task_state_binding = state_binding.clone();
        let language_owned = language.map(ToOwned::to_owned);
        let prefix_mode = GraniteSaaPrefixMode::from_env();
        let prefix_text = granite_saa_chunk_prefix_text(&assembler, prefix_mode);
        let prefix_chars = prefix_text
            .as_deref()
            .map(|text| text.chars().count())
            .unwrap_or(0);
        let chunk_sample_count = chunk_end - chunk_start;
        let max_new_tokens =
            granite_saa_max_new_tokens(&samples[chunk_start..chunk_end], sample_rate);
        let chunk_duration_secs = samples_to_seconds_f64(chunk_sample_count, sample_rate);
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            chunk_start_secs = samples_to_seconds_f64(chunk.start_sample, sample_rate),
            chunk_end_secs = samples_to_seconds_f64(chunk.end_sample, sample_rate),
            chunk_duration_secs,
            max_new_tokens,
            prefix_mode = prefix_mode.as_str(),
            prefix_chars,
            "starting Granite SAA chunk decode"
        );
        let chunk_started = Instant::now();
        let (returned_lease, transcription) = coordinator
            .run_loaded_blocking_stage_with_invocation_paged(
                job,
                execution_contract.clone(),
                task_state_binding,
                WorkUnit::PipelineStage {
                    name: "asr.speaker_attributed.chunk".to_string(),
                    ordinal: idx,
                },
                move |leases| {
                    let chunk_audio = task_samples[chunk_start..chunk_end].to_vec();
                    observation_job.record_materialized_usage(
                        decoded_audio_with_scratch_observation(
                            retained_input_bytes,
                            task_samples.len(),
                            chunk_audio.capacity(),
                        )?,
                    )?;
                    let transcription = granite_saa_transcribe_chunk(
                        &task_model,
                        &chunk_audio,
                        sample_rate,
                        language_owned.as_deref(),
                        prefix_text.as_deref(),
                        max_new_tokens,
                        granite_saa_single_invocation_cache(leases)?,
                    );
                    let steady_usage =
                        decoded_audio_observation(retained_input_bytes, task_samples.len())?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(chunk_audio);
                    Ok((residency_lease, transcription?))
                },
            )
            .await?;
        residency_lease = returned_lease;
        let decode_diagnostics = granite_saa_decode_diagnostics(transcription.diagnostics.as_ref());
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            elapsed_ms = chunk_started.elapsed().as_secs_f64() * 1000.0,
            generated_tokens = ?decode_diagnostics.generated_tokens,
            max_new_tokens = decode_diagnostics.max_new_tokens.unwrap_or(max_new_tokens),
            stop_reason = decode_diagnostics
                .stop_reason
                .as_deref()
                .unwrap_or("unknown"),
            prompt_tokens = ?decode_diagnostics.prompt_tokens,
            prompt_prefix_tokens = ?decode_diagnostics.prompt_prefix_tokens,
            "finished Granite SAA chunk decode"
        );

        if language_out.is_none() {
            language_out = transcription.language.clone();
        }
        if transcription.text.trim().is_empty() {
            warnings.push(format!(
                "Granite SAA chunk {} returned empty text.",
                idx + 1
            ));
        }
        if let Some(warning) = granite_saa_max_token_warning(idx, &decode_diagnostics) {
            tracing::warn!(
                chunk_index = idx + 1,
                warning = warning.as_str(),
                "Granite SAA chunk reached decode token limit"
            );
            warnings.push(warning);
        }
        warnings.extend(assembler.push_chunk_text(transcription.text.as_str(), idx));

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));
    }

    on_progress(granite_saa_complete_progress(&chunks, sample_rate));

    Ok(GraniteSaaLongFormOutput {
        text: assembler.text(),
        language: language_out,
        warnings,
    })
}

fn granite_saa_transcribe_chunk(
    model: &NativeAsrModel,
    audio: &[f32],
    sample_rate: u32,
    language: Option<&str>,
    prefix_text: Option<&str>,
    max_new_tokens: usize,
    cache: &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
) -> Result<NativeAsrTranscription> {
    if let Some(rate) = model.audio_token_rate() {
        // Uniform invocation admission: refuse before spending encoder work
        // on an invocation whose demand provably cannot fit. The family's
        // exact per-step check remains the authority behind this estimate.
        let duration_secs = if sample_rate > 0 {
            audio.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let audio_tokens = (duration_secs * rate).ceil() as usize;
        let prefix_tokens = prefix_text.map_or(0, |text| text.chars().count() / 4);
        let demand = ASR_TEXT_PROMPT_TOKEN_RESERVE
            .saturating_add(audio_tokens)
            .saturating_add(prefix_tokens)
            .saturating_add(max_new_tokens);
        let capacity = cache.capacity_tokens();
        if demand > capacity {
            return Err(Error::InvalidInput(format!(
                "Speaker-attributed ASR invocation needs about {demand} KV tokens (prompt reserve + {audio_tokens} audio + {prefix_tokens} prefix + {max_new_tokens} decode) but the invocation cache has capacity for {capacity}; reduce chunk length or free memory"
            )));
        }
    }
    model.transcribe_granite_speech_task_and_options_physical(
        audio,
        sample_rate,
        language,
        GraniteSpeechTask::SpeakerAttributed,
        prefix_text,
        NativeAsrGenerationOptions {
            max_new_tokens,
            ..NativeAsrGenerationOptions::default()
        },
        cache,
    )
}

fn granite_saa_single_invocation_cache(
    leases: &mut crate::kv::v2::InvocationPagedLeaseSetV2,
) -> Result<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
    let domains = leases.domains().collect::<Vec<_>>();
    let [domain] = domains.as_slice() else {
        return Err(Error::InferenceError(format!(
            "Granite speaker-attributed ASR requires one invocation KV domain, found {}",
            domains.len()
        )));
    };
    leases.cache_mut(*domain)
}

fn granite_saa_processing_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Processing,
        current_chunk: None,
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(0.0),
        total_audio_secs,
        percent: Some(0.0),
    }
}

fn granite_saa_complete_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Complete,
        current_chunk: Some(chunks.len()),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: total_audio_secs,
        total_audio_secs,
        percent: Some(100.0),
    }
}

fn granite_saa_chunk_progress(
    phase: AsrProgressPhase,
    index: usize,
    chunk: &AudioChunk,
    chunks: &[AudioChunk],
    sample_rate: u32,
) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|last| samples_to_seconds_f64(last.end_sample, sample_rate));
    let processed_audio_secs = match phase {
        AsrProgressPhase::ChunkStarted => samples_to_seconds_f64(chunk.start_sample, sample_rate),
        AsrProgressPhase::ChunkFinished => samples_to_seconds_f64(chunk.end_sample, sample_rate),
        AsrProgressPhase::Processing => 0.0,
        AsrProgressPhase::Aligning | AsrProgressPhase::Complete => {
            total_audio_secs.unwrap_or_default()
        }
    };
    let percent = total_audio_secs
        .filter(|total| *total > 0.0)
        .map(|total| ((processed_audio_secs / total) * 100.0).clamp(0.0, 100.0));

    AsrProgress {
        phase,
        current_chunk: Some(index + 1),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(processed_audio_secs),
        total_audio_secs,
        percent,
    }
}

fn samples_to_seconds_f64(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64
    }
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_positive_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

#[cfg(test)]
fn speaker_attributed_asr_result_from_text(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> SpeakerAttributedAsrResult {
    speaker_attributed_asr_result_from_text_with_warnings(
        raw_text,
        language,
        duration_secs,
        min_speakers,
        max_speakers,
        Vec::new(),
    )
}

fn speaker_attributed_asr_result_from_text_with_warnings(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    mut warnings: Vec<String>,
) -> SpeakerAttributedAsrResult {
    let parsed = parse_granite_speech_output(raw_text);
    let mut speakers = HashSet::<String>::new();
    let mut turns = speaker_attributed_asr_turns_from_text(raw_text);

    for turn in &turns {
        let speaker = turn.speaker.trim();
        if speaker != UNKNOWN_SAA_SPEAKER {
            speakers.insert(speaker.to_string());
        }
    }
    if turns.is_empty() && !parsed.text.trim().is_empty() {
        turns.push(SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: parsed.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        });
    }

    let speaker_count = speakers.len();
    if let Some(min_speakers) = min_speakers {
        if speaker_count < min_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), below requested minimum {min_speakers}."
            ));
        }
    }
    if let Some(max_speakers) = max_speakers {
        if speaker_count > max_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), above requested maximum {max_speakers}."
            ));
        }
    }
    if parsed.text.trim().is_empty() {
        warnings.push("Granite SAA returned an empty transcript.".to_string());
    }

    SpeakerAttributedAsrResult {
        text: parsed.text,
        language,
        duration_secs,
        speaker_turns: turns,
        speaker_count,
        status: if warnings.is_empty() {
            SpeakerAttributedAsrStatus::Ready
        } else {
            SpeakerAttributedAsrStatus::Warning
        },
        warnings,
    }
}

fn format_saa_turns(turns: &[SpeakerAttributedAsrTurn]) -> String {
    turns
        .iter()
        .map(format_saa_turn)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_saa_turn(turn: &SpeakerAttributedAsrTurn) -> String {
    format!("[{}]: {}", turn.speaker, turn.text.trim())
}

fn append_with_spacing(target: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if !target.trim().is_empty() && !target.ends_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

fn overlap_prefix_word_count(
    existing: &str,
    incoming: &str,
    min_words: usize,
    max_words: usize,
) -> usize {
    let existing_words = normalized_overlap_words(existing);
    let incoming_words = normalized_overlap_words(incoming);
    let max_words = max_words
        .min(existing_words.len())
        .min(incoming_words.len());
    if max_words < min_words {
        return 0;
    }

    for count in (min_words..=max_words).rev() {
        if existing_words[existing_words.len() - count..] == incoming_words[..count] {
            return count;
        }
    }
    0
}

fn normalized_overlap_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(normalize_overlap_word)
        .collect()
}

fn normalize_overlap_word(word: &str) -> Option<String> {
    let normalized = word
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn drop_prefix_words(text: &str, words_to_drop: usize) -> String {
    text.split_whitespace()
        .skip(words_to_drop)
        .collect::<Vec<_>>()
        .join(" ")
}

fn recent_word_suffix(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut selected = Vec::<&str>::new();
    let mut selected_chars = 0usize;
    for word in text.split_whitespace().rev() {
        let separator_chars = usize::from(!selected.is_empty());
        let word_chars = word.chars().count();
        if selected_chars + separator_chars + word_chars > max_chars {
            break;
        }
        selected.push(word);
        selected_chars += separator_chars + word_chars;
    }
    selected.reverse();
    selected.join(" ")
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
