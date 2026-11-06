//! OpenAI-compatible speech synthesis endpoints.
//!
//! One request is delivered one of three ways, all through the same
//! streaming encoder (`izwi-audio-codec`):
//! - **buffered** (no `stream`, no `stream_format`): the whole file after
//!   generation, with `Content-Length` and statistics headers;
//! - **`stream_format: "audio"`**: the encoded bytes as a chunked body while
//!   generation runs (OpenAI's default delivery);
//! - **SSE** (`stream: true` or `stream_format: "sse"`): base64 deltas of the
//!   same byte stream, as `speech.audio.delta` and Izwi's `audio.chunk`.
//!
//! Every request gets an id (`X-Izwi-Speech-Id`) whose outcome
//! `GET /v1/audio/speech/{id}` returns.

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::Response,
    Json,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::info;

use crate::api::request_context::RequestContext;
use crate::api::saved_voices::resolve_saved_voice_reference;
use crate::api::tts_policy::resolve_tts_output_frames;
use crate::error::{core_api_error, ApiError, ApiJson};
use crate::speech_outcomes::{
    new_speech_id, SpeechCompletion, SpeechMode, SpeechOutcome, SpeechOutcomeGuard, SpeechUsage,
};
use crate::state::AppState;
use izwi_audio_codec::{AudioFormat, CodecError, EncoderOptions, Framing, StreamEncoder};
use izwi_core::{
    parse_tts_model_variant, AudioChunk, GenerationConfig, GenerationRequest, ModelVariant,
    TtsStopReason, WorkloadClass,
};

const SPEECH_RESPONSE_EXPOSED_HEADERS: &str = "X-Generation-Time-Ms, X-Audio-Duration-Secs, X-RTF, X-RTF-Basis, X-Wall-Time-Ms, X-Wall-RTF, X-Tokens-Generated, X-Audio-Sample-Rate, X-Audio-Channels, X-Stop-Reason, X-Output-Truncated, X-Izwi-Tts-Diagnostics, X-Requested-Response-Format, X-Actual-Response-Format, X-Response-Format-Fallback, X-Izwi-Speech-Id, X-SGLang-Omni-Speech-Id, X-Izwi-Streaming-Mode";

/// OpenAI's `instructions` and `input` cap, in characters.
const OPENAI_TEXT_CAP_CHARS: usize = 4096;
/// The rate OpenAI's headerless `pcm` is always at.
const OPENAI_PCM_RATE: u32 = 24_000;

/// The voice a request names: a model speaker (or an uploaded voice) by
/// name, or a saved voice by id, OpenAI's custom-voice object
/// `{"id": "..."}`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum SpeechVoice {
    Name(String),
    Saved { id: String },
}

impl SpeechVoice {
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Name(name) => Some(name),
            Self::Saved { .. } => None,
        }
    }
}

impl From<&str> for SpeechVoice {
    fn from(name: &str) -> Self {
        Self::Name(name.to_string())
    }
}

impl From<String> for SpeechVoice {
    fn from(name: String) -> Self {
        Self::Name(name)
    }
}

/// OpenAI-compatible speech synthesis request.
#[derive(Debug, Default, Deserialize)]
pub struct SpeechRequest {
    /// OpenAI field name for TTS model; an operator alias
    /// (`IZWI_OPENAI_MODEL_ALIASES`) may stand for an Izwi model.
    pub model: String,
    /// OpenAI field name for text input; empty when a `script` gives the
    /// text.
    #[serde(default)]
    pub input: String,
    /// OpenAI-style voice selection: a name, or `{"id"}` for a saved voice.
    #[serde(default)]
    pub voice: Option<SpeechVoice>,
    /// OpenAI response format, or one of Izwi's raw and telephony formats.
    #[serde(default)]
    pub response_format: Option<String>,
    /// Allow a recognized format this build cannot encode (`aac`) to return
    /// WAV bytes instead of a 400.
    #[serde(default)]
    pub allow_format_fallback: Option<bool>,
    /// izwi extension (vLLM-Omni's name): the rate to encode at; the audio
    /// is resampled when the model's rate differs.
    #[serde(default)]
    pub sample_rate: Option<u32>,
    /// izwi extension: target bitrate in bit/s for `mp3` and `opus`.
    #[serde(default)]
    pub bitrate: Option<u32>,
    /// OpenAI speed.
    #[serde(default)]
    pub speed: Option<f32>,
    /// Optional language hint (e.g. "Auto", "English", "Chinese").
    #[serde(default)]
    pub language: Option<String>,
    /// Optional sampling temperature.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Optional max token budget.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Alias for max output tokens in newer APIs.
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    /// Optional top-k sampling for model-specific runtimes.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// If true, stream server-sent events from the same endpoint.
    #[serde(default)]
    pub stream: Option<bool>,
    /// OpenAI's delivery: `audio` (a chunked body of the encoded file) or
    /// `sse` (server-sent events).
    #[serde(default)]
    pub stream_format: Option<String>,
    /// Optional voice design prompt.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Optional reference audio for voice cloning: base64, a data URL, or
    /// an http(s) URL on `IZWI_ALLOWED_MEDIA_DOMAINS`. SGLang-Omni and
    /// vLLM-Omni call it `ref_audio`.
    #[serde(default, alias = "ref_audio")]
    pub reference_audio: Option<String>,
    /// Optional reference transcript for cloning (`ref_text` elsewhere).
    #[serde(default, alias = "ref_text")]
    pub reference_text: Option<String>,
    /// izwi extension: the language the reference audio is spoken in, for
    /// models that clone across languages (Maya 2 Native); absent, the model
    /// infers it from the transcript.
    #[serde(default)]
    pub reference_language: Option<String>,
    /// Optional saved voice identifier resolved server-side.
    #[serde(default)]
    pub saved_voice_id: Option<String>,
    /// izwi extension: reference voices per speaker, for models that voice
    /// several speakers (Fish Audio Drama 3); replaces `reference_audio`,
    /// `reference_text` and `saved_voice_id`.
    #[serde(default)]
    pub references: Vec<crate::api::speech_dialogue::SpeechDialogueReference>,
    /// izwi extension: a dialogue as `{speaker, text}` lines, spoken as
    /// `<|speaker:N|> text` lines; replaces `input`.
    #[serde(default)]
    pub script: Option<Vec<crate::api::speech_dialogue::SpeechScriptLine>>,
    /// izwi extensions: speech sampling (`top_p`, `seed`, `subtalker_*`,
    /// `cfg_scale`, `diffusion_steps`, `repetition_penalty`).
    #[serde(flatten)]
    pub sampling: crate::api::tts_policy::SpeechSamplingRequest,
}

#[derive(Debug, Serialize)]
struct SpeechStreamEvent {
    event: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sequence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_base64: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    is_final: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_generated: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_time_ms: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    audio_duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rtf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// On `audio.done`: `stop` when the model ended speech, `length` when
    /// the output budget cut it.
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_reason: Option<&'static str>,
}

impl SpeechStreamEvent {
    fn new(event: &'static str, request_id: &str) -> Self {
        Self {
            event,
            timing: None,
            request_id: Some(request_id.to_string()),
            sequence: None,
            audio_base64: None,
            sample_count: None,
            is_final: None,
            sample_rate: None,
            audio_format: None,
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
            stop_reason: None,
        }
    }
}

/// One SSE event as JSON, carrying OpenAI's speech-stream shape beside
/// izwi's (TPIPE-11): an `audio.chunk` is also a `speech.audio.delta` with
/// the encoded bytes in `audio`, `audio.done` is also a `speech.audio.done`
/// with `usage`, and `audio.failed` is also an `error` event (SGLang-Omni's
/// name). Existing clients keep reading `event` and `audio_base64`.
fn openai_stream_payload(event: &SpeechStreamEvent) -> String {
    let mut value = serde_json::to_value(event).unwrap_or(serde_json::Value::Null);
    if let Some(object) = value.as_object_mut() {
        match event.event {
            "audio.chunk" => {
                object.insert("type".into(), "speech.audio.delta".into());
                if let Some(audio) = &event.audio_base64 {
                    object.insert("audio".into(), audio.clone().into());
                }
            }
            "audio.done" => {
                object.insert("type".into(), "speech.audio.done".into());
                let usage = SpeechUsage::new(0, event.tokens_generated.unwrap_or(0));
                object.insert(
                    "usage".into(),
                    serde_json::json!({
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                    }),
                );
            }
            "audio.failed" => {
                object.insert("type".into(), "error".into());
            }
            _ => {}
        }
    }
    value.to_string()
}

fn stream_event_queue_capacity() -> usize {
    crate::env_config::tuning().audio_stream_event_queue_capacity
}

/// The format a request asked for, and how it goes out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedSpeechFormat {
    pub(crate) format: AudioFormat,
    /// The name responses report (`pcm` stays `pcm`).
    pub(crate) label: &'static str,
    pub(crate) content_type: &'static str,
    /// OpenAI's `pcm` is 24 kHz whatever the model's rate.
    pub(crate) forced_rate: Option<u32>,
    pub(crate) fallback: Option<String>,
}

/// The speech output a request asked for: format, rate and bitrate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpeechOutput {
    pub(crate) format: ResolvedSpeechFormat,
    pub(crate) sample_rate: Option<u32>,
    pub(crate) bitrate: Option<u32>,
}

impl SpeechOutput {
    /// The output `response_format`, `sample_rate` and `bitrate` describe,
    /// checked before any model work.
    pub(crate) fn resolve(
        response_format: Option<&str>,
        allow_fallback: bool,
        sample_rate: Option<u32>,
        bitrate: Option<u32>,
    ) -> Result<Self, ApiError> {
        let default_format = crate::env_config::tuning()
            .speech_api
            .default_format
            .as_str();
        let requested = response_format
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(default_format);
        let output = Self {
            format: parse_response_format(requested, allow_fallback)?,
            sample_rate,
            bitrate,
        };
        output.check()?;
        Ok(output)
    }

    /// Checks what can be checked before the model's rate is known: a
    /// requested rate exactly, a bitrate's range, and that the format is
    /// built in.
    fn check(&self) -> Result<(), ApiError> {
        let input_rate = self.target_rate().unwrap_or(24_000);
        let mut options = self.options(input_rate, Framing::Stream);
        if self.target_rate().is_none() {
            options.bitrate = None;
        }
        options.validate().map_err(codec_api_error)?;
        if let Some(bitrate) = self.bitrate {
            if !self.format.format.takes_bitrate() {
                return Err(ApiError::invalid_param(
                    "bitrate",
                    format!("bitrate applies to mp3 and opus, not {}", self.format.label),
                ));
            }
            if !(6_000..=510_000).contains(&bitrate) {
                return Err(ApiError::invalid_param(
                    "bitrate",
                    format!("bitrate {bitrate} is out of range (6000 to 510000 bit/s)"),
                ));
            }
        }
        Ok(())
    }

    fn target_rate(&self) -> Option<u32> {
        self.sample_rate.or(self.format.forced_rate)
    }

    pub(crate) fn options(&self, model_rate: u32, framing: Framing) -> EncoderOptions {
        EncoderOptions::new(self.format.format, model_rate)
            .with_output_rate(self.target_rate())
            .with_bitrate(self.bitrate)
            .with_framing(framing)
    }

    pub(crate) fn label(&self) -> &'static str {
        self.format.label
    }

    pub(crate) fn content_type(&self) -> &'static str {
        self.format.content_type
    }

    /// A streaming encoder for audio at `model_rate`.
    pub(crate) fn stream_encoder(
        &self,
        model_rate: u32,
    ) -> Result<Box<dyn StreamEncoder>, ApiError> {
        izwi_audio_codec::stream_encoder(&self.options(model_rate, Framing::Stream))
            .map_err(codec_api_error)
    }

    /// `samples` at `model_rate` as one file, with exact headers.
    pub(crate) fn encode_file(
        &self,
        samples: &[f32],
        model_rate: u32,
    ) -> Result<Vec<u8>, ApiError> {
        izwi_audio_codec::encode_file(&self.options(model_rate, Framing::File), samples)
            .map_err(codec_api_error)
    }

    /// The rate audio from a `model_rate` model goes out at.
    pub(crate) fn output_rate(&self, model_rate: u32) -> u32 {
        self.options(model_rate, Framing::Stream)
            .validate()
            .unwrap_or(model_rate)
    }
}

/// The API error for an encoder error: options and formats the request
/// chose are its fault (400); an encoder failure is the server's (500).
pub(crate) fn codec_api_error(err: CodecError) -> ApiError {
    match err {
        CodecError::Unsupported(message) => ApiError::invalid_param("response_format", message),
        CodecError::InvalidOption(message) => ApiError::bad_request(message),
        CodecError::Encode(message) => {
            ApiError::internal(format!("Audio encoding failed: {message}"))
        }
    }
}

/// How the response is delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpeechDelivery {
    Buffered,
    AudioStream,
    Sse,
}

impl SpeechDelivery {
    const fn mode(self) -> SpeechMode {
        match self {
            Self::Buffered => SpeechMode::Buffered,
            Self::AudioStream => SpeechMode::AudioStream,
            Self::Sse => SpeechMode::Sse,
        }
    }
}

pub async fn speech(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<SpeechRequest>,
) -> Result<Response<Body>, ApiError> {
    let prepared = prepare_speech(&state, ctx.tenant_key(), ctx.correlation_id, req, None).await?;
    run_prepared_speech(state, prepared).await
}

/// A validated speech request, admitted to the outcome store.
pub(crate) struct PreparedSpeech {
    job: SpeechJob,
    delivery: SpeechDelivery,
    guard: SpeechOutcomeGuard,
}

impl PreparedSpeech {
    pub(crate) fn speech_id(&self) -> &str {
        &self.job.speech_id
    }

    pub(crate) fn output(&self) -> &SpeechOutput {
        &self.job.output
    }

    pub(crate) fn is_streamed(&self) -> bool {
        self.delivery != SpeechDelivery::Buffered
    }
}

/// A speech request that passed every check that needs no model work.
pub(crate) struct ValidatedSpeech {
    pub(crate) req: SpeechRequest,
    pub(crate) variant: ModelVariant,
    pub(crate) output: SpeechOutput,
    delivery: SpeechDelivery,
}

/// Validates `req` (everything that needs no model work) and records it in
/// the outcome store; `mode` overrides the delivery's (batch items).
pub(crate) async fn prepare_speech(
    state: &AppState,
    tenant_key: Option<[u8; 32]>,
    correlation_id: String,
    req: SpeechRequest,
    mode: Option<SpeechMode>,
) -> Result<PreparedSpeech, ApiError> {
    let ValidatedSpeech {
        req,
        variant,
        output,
        delivery,
    } = validate_speech(state, req).await?;
    let speech_id = new_speech_id();
    state.speech_outcomes.begin(
        &speech_id,
        tenant_key,
        variant.dir_name(),
        mode.unwrap_or(delivery.mode()),
        output.label(),
    );
    let guard = SpeechOutcomeGuard::new(state.speech_outcomes.clone(), speech_id.clone());
    Ok(PreparedSpeech {
        job: SpeechJob {
            req,
            variant,
            output,
            correlation_id,
            tenant_key,
            speech_id,
        },
        delivery,
        guard,
    })
}

/// Every check of a speech request that needs no model work: the model and
/// its aliases, voices (uploaded names, saved ids, reference URLs), the
/// model's controls and the output format. The text-in socket validates its
/// session configuration through this too.
pub(crate) async fn validate_speech(
    state: &AppState,
    req: SpeechRequest,
) -> Result<ValidatedSpeech, ApiError> {
    let mut req = normalize_speech_request(req)?;
    if let Some(text) =
        crate::api::speech_dialogue::script_text(Some(&req.input), req.script.as_deref(), "input")?
    {
        req.input = text;
    }
    if req.input.trim().is_empty() {
        return Err(ApiError::bad_request(
            "Missing required `input` field (or a `script` to speak).",
        ));
    }
    crate::api::tts_policy::check_tts_input_length(&req.input)?;
    check_instructions_length(req.instructions.as_deref())?;
    req.model = crate::api::openai::model_aliases::resolve_model_alias(&req.model).to_string();
    let variant = parse_tts_model_variant(&req.model)
        .map_err(|err| ApiError::bad_request(format!("Unsupported TTS model: {}", err)))?;
    crate::api::speech_dialogue::check_dialogue_request(variant, speech_dialogue(&req))?;
    resolve_uploaded_voice_name(state, &mut req, variant).await?;
    fetch_remote_reference(&mut req).await?;
    let req = resolve_saved_voice_request(state, req).await?;
    info!("OpenAI speech request: {} bytes", req.input.len());
    validate_speech_voice_contract(&req, variant)?;
    crate::api::tts_policy::validate_speech_speed(variant, req.speed)?;
    crate::api::tts_policy::validate_speech_controls(variant, &req.sampling)?;
    crate::api::tts_policy::validate_speech_language(variant, req.language.as_deref())?;
    crate::api::tts_policy::validate_speech_reference_language(
        variant,
        req.reference_language.as_deref(),
    )?;
    crate::api::tts_policy::validate_speech_inline_tags(variant, &req.input)?;
    let delivery = resolve_delivery(&req)?;
    crate::api::tts_policy::validate_speech_streaming(
        variant,
        delivery != SpeechDelivery::Buffered,
    )?;
    let output = SpeechOutput::resolve(
        req.response_format.as_deref(),
        req.allow_format_fallback.unwrap_or(false),
        req.sample_rate,
        req.bitrate,
    )?;

    Ok(ValidatedSpeech {
        req,
        variant,
        output,
        delivery,
    })
}

/// Runs a prepared request in its delivery mode.
pub(crate) async fn run_prepared_speech(
    state: AppState,
    prepared: PreparedSpeech,
) -> Result<Response<Body>, ApiError> {
    let PreparedSpeech {
        job: request,
        delivery,
        guard,
    } = prepared;
    let variant = request.variant;
    match delivery {
        SpeechDelivery::Sse => stream_speech_sse(state, request, guard).await,
        SpeechDelivery::AudioStream => stream_speech_audio(state, request, guard).await,
        SpeechDelivery::Buffered if variant.plans_own_speech_segments() => {
            fish_file_speech(state, request, guard).await
        }
        SpeechDelivery::Buffered => buffered_speech(state, request, guard).await,
    }
}

/// One admitted speech request.
struct SpeechJob {
    req: SpeechRequest,
    variant: ModelVariant,
    output: SpeechOutput,
    correlation_id: String,
    tenant_key: Option<[u8; 32]>,
    speech_id: String,
}

/// Records `error` as the request's outcome, and returns it.
fn failed(state: &AppState, speech_id: &str, error: ApiError) -> ApiError {
    state.speech_outcomes.fail(
        speech_id,
        &error.message,
        if error.status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        },
        error.detail.and_then(|detail| detail.code),
    );
    error
}

async fn buffered_speech(
    state: AppState,
    job: SpeechJob,
    guard: SpeechOutcomeGuard,
) -> Result<Response<Body>, ApiError> {
    let SpeechJob {
        req,
        variant,
        output,
        correlation_id,
        tenant_key,
        speech_id,
    } = job;
    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;
    let request_started = std::time::Instant::now();
    state
        .runtime
        .load_model(variant)
        .await
        .map_err(|err| failed(&state, &speech_id, core_api_error(err)))?;

    // No wall clock wraps active generation (TPIPE-6): it ends on its own,
    // on an error, or when the client disconnects and drops this future,
    // which cancels the engine work.
    let result = {
        let mut runtime_context = permit.runtime_context();
        runtime_context.tenant_key = tenant_key;
        let mut gen_request = build_generation_request(&req, correlation_id, false, variant)
            .with_runtime_context(runtime_context);
        // Text one automatic request cannot finish is split by long form,
        // as on the first-party route, instead of being cut at the cap.
        crate::api::tts_long_form::defer_automatic_budget_to_long_form(
            &mut gen_request,
            variant,
            req.max_output_tokens.or(req.max_tokens),
        );
        crate::api::tts_long_form::generate_long_form_tts(&state.runtime, variant, gen_request)
            .await
            .map_err(|err| failed(&state, &speech_id, core_api_error(err)))?
    };

    let sample_rate = result.sample_rate;
    let samples = result.samples.clone();
    let encode_output = output.clone();
    let audio_bytes =
        tokio::task::spawn_blocking(move || encode_output.encode_file(&samples, sample_rate))
            .await
            .map_err(|e| {
                failed(
                    &state,
                    &speech_id,
                    ApiError::internal(format!("Audio encoding failed: {e}")),
                )
            })?
            .map_err(|err| failed(&state, &speech_id, err))?;

    let output_rate = output.output_rate(sample_rate);
    let duration_secs = result.duration_secs();
    let generation_time_ms = result.total_time_ms;
    let rtf = result.rtf();
    let tokens_generated = result.total_tokens;
    // Two RTF bases, both reported (TPIPE-9): X-RTF over engine execution
    // (summed across long-form segments) and X-Wall-RTF over the request's
    // wall time from admission, as the streams report it.
    let wall_ms = request_started.elapsed().as_secs_f64() * 1000.0;
    let wall_rtf = if duration_secs > 0.0 {
        wall_ms / 1000.0 / f64::from(duration_secs)
    } else {
        0.0
    };

    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, output.content_type())
        .header("X-Generation-Time-Ms", format!("{:.1}", generation_time_ms))
        .header("X-Audio-Duration-Secs", format!("{:.2}", duration_secs))
        .header("X-RTF", format!("{:.3}", rtf))
        .header("X-RTF-Basis", "engine_execution")
        .header("X-Wall-Time-Ms", format!("{wall_ms:.1}"))
        .header("X-Wall-RTF", format!("{wall_rtf:.3}"))
        .header("X-Tokens-Generated", tokens_generated.to_string())
        .header("X-Audio-Sample-Rate", output_rate.to_string())
        .header("X-Audio-Channels", "1")
        .header("X-Izwi-Speech-Id", speech_id.as_str())
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        )
        .header(
            "X-Requested-Response-Format",
            req.response_format
                .as_deref()
                .unwrap_or(output.label())
                .to_ascii_lowercase(),
        )
        .header("X-Actual-Response-Format", output.label());
    if let Some(fallback) = output.format.fallback.as_deref() {
        builder = builder
            .header("X-Response-Format-Fallback", fallback)
            .header(
                "Warning",
                format!(
                    "299 Izwi \"Requested response_format returned {}; fallback was explicitly enabled\"",
                    output.label()
                ),
            );
    }
    if let Some(stop_reason) = result.stop_reason {
        builder = builder
            .header("X-Stop-Reason", stop_reason.as_str())
            .header("X-Output-Truncated", stop_reason.truncated().to_string());
    }
    if let Some(diagnostics) = result.diagnostics.as_ref() {
        builder = builder.header("X-Izwi-Tts-Diagnostics", diagnostics.to_string());
    }
    state.speech_outcomes.complete(
        &speech_id,
        SpeechCompletion {
            sample_rate: Some(output_rate),
            audio_duration_secs: Some(duration_secs),
            finish_reason: Some(finish_reason(result.stop_reason)),
            usage: Some(SpeechUsage::new(0, tokens_generated)),
            timing: Some(serde_json::json!({
                "generation_time_ms": generation_time_ms,
                "wall_time_ms": wall_ms,
                "rtf": rtf,
            })),
        },
    );
    drop(guard);
    Ok(builder.body(Body::from(audio_bytes)).unwrap())
}

fn finish_reason(stop_reason: Option<TtsStopReason>) -> &'static str {
    match stop_reason {
        Some(reason) if reason.truncated() => "length",
        _ => "stop",
    }
}

/// Where a Fish family's file response is assembled: WAV in the speech
/// spool (exact sizes), any other format through a streaming encoder into a
/// raw spool file.
enum FishSpool {
    Wav {
        spool: crate::speech_history_store::SpeechWavSpool,
        /// Boxed: the FFT resampler is far larger than the encoder handle.
        resampler: Option<Box<izwi_audio_codec::StreamResampler>>,
    },
    Encoded(Box<dyn StreamEncoder>),
}

/// Synchronous requests retain cancellation with their HTTP future. Audio is
/// spooled incrementally; the response owns the finalized file until EOF/drop.
async fn fish_file_speech(
    state: AppState,
    job: SpeechJob,
    guard: SpeechOutcomeGuard,
) -> Result<Response<Body>, ApiError> {
    use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
    let SpeechJob {
        req,
        variant,
        output,
        correlation_id,
        tenant_key,
        speech_id,
    } = job;
    let mut request = build_generation_request(&req, correlation_id, true, variant);
    request.runtime_context.tenant_key = tenant_key;
    // Validate admission before model load or output allocation.
    crate::api::tts_long_form::SpeechTextPlan::for_variant(
        variant,
        &req.input,
        request.config.options.max_tokens,
    )
    .map_err(|err| failed(&state, &speech_id, core_api_error(err)))?;
    state
        .runtime
        .load_model(variant)
        .await
        .map_err(|err| failed(&state, &speech_id, core_api_error(err)))?;
    let started = Instant::now();
    let (sender, mut receiver) = mpsc::channel::<AudioChunk>(2);
    let collect_output = output.clone();
    let collect = async {
        let output = collect_output;
        let raw_owner = crate::speech_spool::new_speech_tempfile()
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut raw = tokio::fs::File::from_std(
            raw_owner
                .reopen()
                .map_err(|e| ApiError::internal(e.to_string()))?,
        );
        let mut reservation = crate::speech_resource_budget::spool_budget()
            .reserve(0)
            .map_err(|e| ApiError::internal(e.to_string()))?;
        let mut spool: Option<FishSpool> = None;
        let mut sample_rate = None;
        let mut samples = 0u64;
        let mut statistics = None;
        while let Some(chunk) = receiver.recv().await {
            accumulate_stream_statistics(&mut statistics, &chunk);
            if chunk.samples.is_empty() {
                continue;
            }
            let rate = chunk.sample_rate_or(44_100);
            if sample_rate.is_some_and(|previous| previous != rate) {
                return Err(ApiError::internal(
                    "Speech sample rate changed between segments",
                ));
            }
            sample_rate = Some(rate);
            samples = samples
                .checked_add(chunk.samples.len() as u64)
                .ok_or_else(|| ApiError::internal("Speech sample count overflow"))?;
            if spool.is_none() {
                spool = Some(open_fish_spool(&output, rate)?);
            }
            match spool.as_mut().expect("opened") {
                FishSpool::Wav { spool, resampler } => {
                    let pcm = match resampler.as_mut() {
                        Some(resampler) => {
                            resampler.push(&chunk.samples).map_err(codec_api_error)?
                        }
                        None => chunk.samples.clone(),
                    };
                    spool
                        .append_pcm(&izwi_audio_codec::pcm16_le(&pcm))
                        .await
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                }
                FishSpool::Encoded(encoder) => {
                    let bytes = encoder.push(&chunk.samples).map_err(codec_api_error)?;
                    reservation
                        .grow(bytes.len())
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                    raw.write_all(&bytes)
                        .await
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                }
            }
        }
        let rate =
            sample_rate.ok_or_else(|| ApiError::internal("Speech generation produced no audio"))?;
        let artifact = match spool.take() {
            Some(FishSpool::Wav {
                mut spool,
                mut resampler,
            }) => {
                if let Some(resampler) = resampler.as_mut() {
                    let tail = resampler.finish().map_err(codec_api_error)?;
                    spool
                        .append_pcm(&izwi_audio_codec::pcm16_le(&tail))
                        .await
                        .map_err(|e| ApiError::internal(e.to_string()))?;
                }
                Some(
                    spool
                        .finish_file()
                        .await
                        .map_err(|e| ApiError::internal(e.to_string()))?,
                )
            }
            Some(FishSpool::Encoded(mut encoder)) => {
                let tail = encoder.finish().map_err(codec_api_error)?;
                reservation
                    .grow(tail.len())
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                raw.write_all(&tail)
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?;
                None
            }
            None => None,
        };
        let (mut file, length) = if let Some(wav) = artifact.as_ref() {
            (
                wav.open()
                    .await
                    .map_err(|e| ApiError::internal(e.to_string()))?,
                wav.len(),
            )
        } else {
            raw.flush()
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            raw.seek(std::io::SeekFrom::Start(0))
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?;
            let length = raw
                .metadata()
                .await
                .map_err(|e| ApiError::internal(e.to_string()))?
                .len();
            (raw, length)
        };
        let stream = async_stream::try_stream! {
            // Capture owners before polling: dropping even an unpolled body releases disk.
            let _owners = (artifact, raw_owner, reservation);
            let mut buffer = vec![0u8; 64 * 1024];
            loop {
                let count = file.read(&mut buffer).await?;
                if count == 0 { break; }
                yield bytes::Bytes::copy_from_slice(&buffer[..count]);
            }
        };
        let body = Body::from_stream(Box::pin(stream)
            as std::pin::Pin<
                Box<dyn futures::Stream<Item = Result<bytes::Bytes, std::io::Error>> + Send>,
            >);
        Ok::<_, ApiError>((body, length, rate, samples, statistics))
    };
    let generation = async {
        crate::api::tts_long_form::generate_speech_plan_stream(
            &state,
            variant,
            request,
            sender,
            WorkloadClass::Interactive,
        )
        .await
        .map_err(core_api_error)
    };
    // The plan runs until it completes, fails, stalls (per-segment progress
    // deadline) or the client disconnects; a fixed wall clock failed long
    // narrations with a 500 after minutes of work (TPIPE-6).
    let (_, (body, length, rate, samples, statistics)) =
        tokio::try_join!(generation, collect).map_err(|err| failed(&state, &speech_id, err))?;
    let duration = samples as f64 / f64::from(rate);
    let elapsed = started.elapsed().as_secs_f64();
    let output_rate = output.output_rate(rate);
    let mut builder = Response::builder()
        .header(header::CONTENT_TYPE, output.content_type())
        .header(header::CONTENT_LENGTH, length)
        .header("X-Audio-Sample-Rate", output_rate)
        .header("X-Audio-Channels", "1")
        .header("X-Generation-Time-Ms", format!("{:.1}", elapsed * 1000.0))
        .header("X-Audio-Duration-Secs", format!("{duration:.2}"))
        .header("X-RTF", format!("{:.3}", elapsed / duration))
        .header("X-Izwi-Speech-Id", speech_id.as_str())
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        )
        .header(
            "X-Requested-Response-Format",
            req.response_format.as_deref().unwrap_or(output.label()),
        )
        .header("X-Actual-Response-Format", output.label());
    let mut tokens = 0;
    let mut stop = None;
    if let Some(stats) = statistics {
        tokens = stats.tokens_generated;
        stop = stats.stop_reason;
        builder = builder.header("X-Tokens-Generated", stats.tokens_generated);
        if let Some(stop_reason) = stats.stop_reason {
            builder = builder
                .header("X-Stop-Reason", stop_reason.as_str())
                .header("X-Output-Truncated", stop_reason.truncated().to_string());
        }
    }
    if let Some(fallback) = output.format.fallback.as_deref() {
        builder = builder.header("X-Response-Format-Fallback", fallback);
    }
    state.speech_outcomes.complete(
        &speech_id,
        SpeechCompletion {
            sample_rate: Some(output_rate),
            audio_duration_secs: Some(duration as f32),
            finish_reason: Some(finish_reason(stop)),
            usage: Some(SpeechUsage::new(0, tokens)),
            timing: Some(serde_json::json!({ "generation_time_ms": elapsed * 1000.0 })),
        },
    );
    drop(guard);
    builder
        .body(body)
        .map_err(|e| ApiError::internal(e.to_string()))
}

fn open_fish_spool(output: &SpeechOutput, rate: u32) -> Result<FishSpool, ApiError> {
    if output.format.format != AudioFormat::Wav {
        return Ok(FishSpool::Encoded(output.stream_encoder(rate)?));
    }
    let output_rate = output.output_rate(rate);
    let resampler = (output_rate != rate)
        .then(|| izwi_audio_codec::StreamResampler::new(rate, output_rate).map(Box::new))
        .transpose()
        .map_err(codec_api_error)?;
    let spool =
        crate::speech_history_store::SpeechWavSpool::new(output_rate, u32::MAX as usize - 44)
            .map_err(|e| ApiError::internal(e.to_string()))?;
    Ok(FishSpool::Wav { spool, resampler })
}

/// `voice: "<name>"` names an uploaded voice (`/v1/audio/voices`) when the
/// model clones from a reference and has no built-in voices of its own, and
/// the request gives no other reference: the way SGLang-Omni and vLLM-Omni
/// clients speak with their uploads. A model with built-in voices always
/// reads the name as its own voice.
async fn resolve_uploaded_voice_name(
    state: &AppState,
    req: &mut SpeechRequest,
    variant: ModelVariant,
) -> Result<(), ApiError> {
    let Some(name) = req.voice.as_ref().and_then(SpeechVoice::name) else {
        return Ok(());
    };
    let clones_without_builtins = variant.speech_capabilities().is_some_and(|capabilities| {
        capabilities.supports_reference_voice && !capabilities.supports_builtin_voices
    });
    if !clones_without_builtins
        || req.saved_voice_id.is_some()
        || req.reference_audio.is_some()
        || !req.references.is_empty()
    {
        return Ok(());
    }
    if let Some(voice) = state
        .saved_voice_store
        .find_voice_by_name(name)
        .await
        .map_err(|err| crate::error::store_api_error(err, "Saved voice storage error"))?
    {
        req.saved_voice_id = Some(voice.id);
    }
    Ok(())
}

/// A `reference_audio` URL fetched under the media policy
/// (`api/remote_audio.rs`), as base64 for the model.
async fn fetch_remote_reference(req: &mut SpeechRequest) -> Result<(), ApiError> {
    let Some(url) = req
        .reference_audio
        .as_deref()
        .filter(|value| crate::api::remote_audio::is_remote_audio_url(value))
    else {
        return Ok(());
    };
    let bytes = crate::api::remote_audio::fetch_reference_audio(url).await?;
    req.reference_audio = Some(base64::engine::general_purpose::STANDARD.encode(bytes));
    Ok(())
}

fn normalize_speech_request(mut req: SpeechRequest) -> Result<SpeechRequest, ApiError> {
    req.voice = match req.voice.take() {
        Some(SpeechVoice::Name(name)) => {
            normalize_optional_trimmed(Some(name)).map(SpeechVoice::Name)
        }
        Some(SpeechVoice::Saved { id }) => {
            let id = id.trim().to_string();
            if id.is_empty() {
                return Err(ApiError::invalid_param("voice", "`voice.id` is empty"));
            }
            // OpenAI's custom-voice object names a saved voice.
            match normalize_optional_trimmed(req.saved_voice_id.take()) {
                Some(saved) if saved != id => {
                    return Err(ApiError::bad_request(
                        "`voice.id` and `saved_voice_id` name different voices; send one.",
                    ))
                }
                _ => {}
            }
            req.saved_voice_id = Some(id.clone());
            Some(SpeechVoice::Saved { id })
        }
        None => None,
    };
    req.response_format = normalize_optional_trimmed(req.response_format);
    req.language = normalize_optional_trimmed(req.language);
    req.instructions = normalize_optional_trimmed(req.instructions);
    req.reference_audio = normalize_optional_trimmed(req.reference_audio);
    req.reference_text = normalize_optional_trimmed(req.reference_text);
    req.reference_language = normalize_optional_trimmed(req.reference_language);
    req.saved_voice_id = normalize_optional_trimmed(req.saved_voice_id);
    req.stream_format = normalize_optional_trimmed(req.stream_format);
    crate::api::speech_dialogue::normalize_references(&mut req.references);
    Ok(req)
}

/// OpenAI caps `instructions` at 4096 characters.
fn check_instructions_length(instructions: Option<&str>) -> Result<(), ApiError> {
    match instructions {
        Some(text) if text.chars().count() > OPENAI_TEXT_CAP_CHARS => Err(ApiError::invalid_param(
            "instructions",
            format!("`instructions` is longer than {OPENAI_TEXT_CAP_CHARS} characters"),
        )),
        _ => Ok(()),
    }
}

/// The dialogue fields of `req`, as the dialogue checks read them.
fn speech_dialogue(req: &SpeechRequest) -> crate::api::speech_dialogue::SpeechDialogue<'_> {
    let present = |value: Option<&str>| value.is_some_and(has_non_empty_text);
    crate::api::speech_dialogue::SpeechDialogue {
        single_reference: present(req.reference_audio.as_deref())
            || present(req.reference_text.as_deref())
            || present(req.saved_voice_id.as_deref()),
        references: &req.references,
        script: req.script.as_deref(),
    }
}

async fn resolve_saved_voice_request(
    state: &AppState,
    mut req: SpeechRequest,
) -> Result<SpeechRequest, ApiError> {
    crate::api::speech_dialogue::resolve_saved_voices(state, &mut req.references).await?;
    let Some(saved_voice_id) = req.saved_voice_id.as_deref() else {
        return Ok(req);
    };

    let has_direct_reference_audio = req
        .reference_audio
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    let has_direct_reference_text = req
        .reference_text
        .as_deref()
        .map(has_non_empty_text)
        .unwrap_or(false);
    if has_direct_reference_audio || has_direct_reference_text {
        return Err(ApiError::bad_request(
            "Use either `saved_voice_id` or direct `reference_audio`/`reference_text`, not both.",
        ));
    }

    let saved_voice = resolve_saved_voice_reference(state, saved_voice_id).await?;
    req.saved_voice_id = Some(saved_voice.voice_id);
    req.reference_audio = Some(saved_voice.reference_audio_base64);
    req.reference_text = saved_voice.reference_text;
    // The voice's language serves a model that reads one; a request's own
    // `reference_language` wins.
    if req.reference_language.is_none()
        && parse_tts_model_variant(&req.model)
            .is_ok_and(crate::api::tts_policy::takes_reference_language)
    {
        req.reference_language = saved_voice.reference_language;
    }
    Ok(req)
}

fn validate_speech_voice_contract(
    req: &SpeechRequest,
    variant: ModelVariant,
) -> Result<(), ApiError> {
    let present = |value: Option<&str>| value.map(has_non_empty_text).unwrap_or(false);
    // OpenAI requires `voice`, so a client sends one even to a model that
    // speaks only with a reference; it is ignored there, not refused.
    let speaker = present(req.voice.as_ref().and_then(SpeechVoice::name))
        && variant
            .speech_capabilities()
            .is_some_and(|capabilities| capabilities.supports_builtin_voices);
    // Caller text holding the model's control tokens would rewrite its
    // prompt.
    crate::api::tts_policy::validate_speech_control_tokens(variant, "input", &req.input)?;
    if let Some(reference_text) = req.reference_text.as_deref() {
        crate::api::tts_policy::validate_speech_control_tokens(
            variant,
            "reference_text",
            reference_text,
        )?;
    }
    // Each dialogue reference's transcript is checked on its own; the voice
    // contract then reads them as the request's reference.
    crate::api::speech_dialogue::check_dialogue_references(variant, &req.references, &req.input)?;
    let dialogue = !req.references.is_empty();
    crate::api::tts_policy::validate_speech_voice(
        variant,
        crate::api::tts_policy::SpeechVoiceRequest {
            speaker,
            reference_audio: dialogue || present(req.reference_audio.as_deref()),
            reference_text: dialogue || present(req.reference_text.as_deref()),
            saved_voice: present(req.saved_voice_id.as_deref()),
            voice_description: present(req.instructions.as_deref()),
        },
    )
}

/// What a speech stream's pump hands the response it feeds.
#[derive(Debug)]
enum PumpEvent {
    /// The first audio is about to go out, at this (output) rate.
    Started {
        sample_rate: u32,
    },
    /// The next encoded bytes (possibly none: codecs work in frames).
    Audio {
        bytes: Vec<u8>,
        request_id: String,
        sequence: usize,
        sample_count: usize,
        is_final: bool,
        sample_rate: u32,
    },
    Done(Box<StreamSummary>),
    Failed {
        error: ApiError,
    },
}

#[derive(Debug)]
struct StreamSummary {
    generation_time_ms: f32,
    tokens_generated: Option<usize>,
    execution_time_ms: Option<f32>,
    audio_duration_secs: f32,
    rtf: f32,
    first_pcm_ms: Option<f64>,
    last_pcm_ms: Option<f64>,
    total_samples: usize,
    sample_rate: u32,
    stop_reason: Option<TtsStopReason>,
    planned_segment_count: usize,
    token_unit: &'static str,
}

impl StreamSummary {
    fn timing(&self) -> serde_json::Value {
        let audio = self.audio_duration_secs;
        serde_json::json!({
            "generation_time_basis": "post_admission_stream_wall",
            "request_timing_basis": "stream_handler_entry_after_request_validation",
            "execution_time_ms": self.execution_time_ms,
            "execution_rtf": self.execution_time_ms.filter(|_| audio > 0.0)
                .map(|ms| ms / 1000.0 / audio),
            "first_pcm_ms": self.first_pcm_ms,
            "request_to_last_pcm_ms": self.last_pcm_ms,
            "request_to_last_pcm_rtf": self.last_pcm_ms.filter(|_| audio > 0.0)
                .map(|ms| ms / 1000.0 / f64::from(audio)),
            "pcm_sample_count": self.total_samples,
            "token_unit": self.token_unit,
            "planned_segment_count": self.planned_segment_count,
        })
    }

    fn completion(&self) -> SpeechCompletion {
        SpeechCompletion {
            sample_rate: Some(self.sample_rate),
            audio_duration_secs: Some(self.audio_duration_secs),
            finish_reason: Some(finish_reason(self.stop_reason)),
            usage: Some(SpeechUsage::new(0, self.tokens_generated.unwrap_or(0))),
            timing: Some(self.timing()),
        }
    }
}

/// Send one pump event, failing a reader that has stalled past the stall
/// deadline (`IZWI_AUDIO_STREAM_STALL_TIMEOUT_SECS`, 30 s), so a connected
/// but stuck client cannot pin the generation and its permit forever
/// (TPIPE-13).
async fn send_pump_event(tx: &mpsc::Sender<PumpEvent>, event: PumpEvent) -> Result<(), ()> {
    let stall_timeout = crate::env_config::tuning().audio_stream_stall_timeout;
    tokio::time::timeout(stall_timeout, tx.send(event))
        .await
        .map_err(|_| ())?
        .map_err(|_| ())
}

/// A speech stream ready to run: its engine request and the plan's size.
struct SpeechPump {
    state: AppState,
    variant: ModelVariant,
    output: SpeechOutput,
    gen_request: GenerationRequest,
    planned_segment_count: usize,
    tenant_key: Option<[u8; 32]>,
    started: Instant,
}

impl SpeechPump {
    fn prepare(state: &AppState, job: &SpeechJob) -> Result<Self, ApiError> {
        let started = Instant::now();
        let variant = job.variant;
        let req = &job.req;
        let mut gen_request =
            build_generation_request(req, job.correlation_id.clone(), true, variant);
        // Long text streams as long-form segments, as a final-only request
        // splits.
        crate::api::tts_long_form::defer_automatic_budget_to_long_form(
            &mut gen_request,
            variant,
            req.max_output_tokens.or(req.max_tokens),
        );
        let planned_segment_count = if variant.plans_own_speech_segments() {
            crate::api::tts_long_form::SpeechTextPlan::for_variant(
                variant,
                &req.input,
                gen_request.config.options.max_tokens,
            )
            .map_err(core_api_error)?
            .segments
            .len()
        } else {
            crate::api::tts_long_form::expand_generation_requests_for_long_form(
                &gen_request,
                variant,
            )
            .len()
        };
        Ok(Self {
            state: state.clone(),
            variant,
            output: job.output.clone(),
            gen_request,
            planned_segment_count,
            tenant_key: job.tenant_key,
            started,
        })
    }

    /// Runs admission, model load and generation, encoding the audio and
    /// handing it to `tx` until the stream ends or the reader goes away.
    fn spawn(self, tx: mpsc::Sender<PumpEvent>) {
        tokio::spawn(self.run(tx));
    }

    async fn run(self, tx: mpsc::Sender<PumpEvent>) {
        let SpeechPump {
            state,
            variant,
            output,
            mut gen_request,
            planned_segment_count,
            tenant_key,
            started,
        } = self;
        let stream_request_id = gen_request.id.clone();
        let fail = |error: ApiError| PumpEvent::Failed { error };
        let engine = state.runtime.clone();
        let permit = match tokio::select! {
            _ = tx.closed() => return,
            result = state.acquire_owned_workload_permit(WorkloadClass::Streaming) => result,
        } {
            Ok(permit) => permit,
            Err(_) => {
                let _ = send_pump_event(
                    &tx,
                    fail(ApiError::of(
                        crate::error::ErrorClass::Unavailable,
                        "Server is shutting down",
                    )),
                )
                .await;
                return;
            }
        };
        let mut runtime_context = permit.runtime_context();
        runtime_context.tenant_key = tenant_key;
        gen_request = gen_request.with_runtime_context(runtime_context);
        let load_result = tokio::select! {
            _ = tx.closed() => return,
            result = engine.load_model(variant) => result,
        };
        if let Err(err) = load_result {
            let _ = send_pump_event(&tx, fail(core_api_error(err))).await;
            return;
        }

        let fallback_sample_rate = engine.sample_rate().await;
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<AudioChunk>(32);
        let generation_engine = engine.clone();
        let retained_permit = if variant.plans_own_speech_segments() {
            drop(permit);
            None
        } else {
            Some(permit)
        };
        let admission_state = state.clone();
        let generation_task = tokio::spawn(async move {
            let _permit = retained_permit;
            if variant.plans_own_speech_segments() {
                crate::api::tts_long_form::generate_speech_plan_stream(
                    &admission_state,
                    variant,
                    gen_request,
                    chunk_tx,
                    WorkloadClass::Streaming,
                )
                .await
            } else {
                crate::api::tts_long_form::generate_long_form_stream(
                    &generation_engine,
                    variant,
                    gen_request,
                    chunk_tx,
                    fallback_sample_rate,
                )
                .await
            }
        });

        let mut encoder: Option<Box<dyn StreamEncoder>> = None;
        let mut source_rate = None;
        let mut total_samples = 0usize;
        let mut terminal_statistics = None;
        let mut first_pcm_ms = None;
        let mut last_pcm_ms = None;
        let mut audio_duration_secs = 0.0f32;
        let stream_started = Instant::now();
        let mut client_closed = false;
        let mut stream_error = None;
        while let Some(chunk) = next_stream_chunk(&mut chunk_rx, &tx).await {
            // A successful runtime terminal marker may contain statistics only.
            accumulate_stream_statistics(&mut terminal_statistics, &chunk);
            if chunk.samples.is_empty() {
                continue;
            }
            let chunk_rate = chunk.sample_rate_or(fallback_sample_rate).max(1);
            if encoder.is_none() {
                match output.stream_encoder(chunk_rate) {
                    Ok(built) => {
                        let sample_rate = built.output_rate();
                        encoder = Some(built);
                        source_rate = Some(chunk_rate);
                        if send_pump_event(&tx, PumpEvent::Started { sample_rate })
                            .await
                            .is_err()
                        {
                            client_closed = true;
                            break;
                        }
                    }
                    Err(error) => {
                        stream_error = Some(error);
                        break;
                    }
                }
            } else if source_rate != Some(chunk_rate) {
                stream_error = Some(ApiError::internal(format!(
                    "TTS audio changed sample rate mid-stream ({} Hz to {chunk_rate} Hz)",
                    source_rate.unwrap_or_default()
                )));
                break;
            }
            let pcm_ms = started.elapsed().as_secs_f64() * 1000.0;
            first_pcm_ms.get_or_insert(pcm_ms);
            last_pcm_ms = Some(pcm_ms);
            total_samples += chunk.samples.len();
            audio_duration_secs += chunk.samples.len() as f32 / chunk_rate as f32;
            let encoder = encoder.as_mut().expect("built above");
            let bytes = match encoder.push(&chunk.samples) {
                Ok(bytes) => bytes,
                Err(err) => {
                    stream_error = Some(codec_api_error(err));
                    break;
                }
            };
            if bytes.is_empty() {
                continue;
            }
            let event = PumpEvent::Audio {
                bytes,
                request_id: chunk.request_id.clone(),
                sequence: chunk.sequence,
                sample_count: chunk.samples.len(),
                is_final: false,
                sample_rate: encoder.output_rate(),
            };
            if send_pump_event(&tx, event).await.is_err() {
                client_closed = true;
                break;
            }
        }

        drop(chunk_rx);
        if client_closed || tx.is_closed() {
            generation_task.abort();
            let _ = generation_task.await;
            return;
        }
        if let Some(error) = stream_error {
            generation_task.abort();
            let _ = generation_task.await;
            let _ = send_pump_event(&tx, fail(error)).await;
            return;
        }

        match generation_task.await {
            Ok(Ok(())) => {
                // A stream with no audio still produces a valid (empty) file.
                let encoder = match encoder {
                    Some(encoder) => Ok(encoder),
                    None => output.stream_encoder(fallback_sample_rate),
                };
                let mut encoder = match encoder {
                    Ok(encoder) => encoder,
                    Err(error) => {
                        let _ = send_pump_event(&tx, fail(error)).await;
                        return;
                    }
                };
                if total_samples == 0
                    && send_pump_event(
                        &tx,
                        PumpEvent::Started {
                            sample_rate: encoder.output_rate(),
                        },
                    )
                    .await
                    .is_err()
                {
                    return;
                }
                let tail = match encoder.finish() {
                    Ok(tail) => tail,
                    Err(err) => {
                        let _ = send_pump_event(&tx, fail(codec_api_error(err))).await;
                        return;
                    }
                };
                let output_rate = encoder.output_rate();
                if send_pump_event(
                    &tx,
                    PumpEvent::Audio {
                        bytes: tail,
                        request_id: stream_request_id.clone(),
                        sequence: usize::MAX,
                        sample_count: 0,
                        is_final: true,
                        sample_rate: output_rate,
                    },
                )
                .await
                .is_err()
                {
                    return;
                }
                let generation_time_ms = stream_started.elapsed().as_secs_f32() * 1000.0;
                let summary = StreamSummary {
                    generation_time_ms,
                    tokens_generated: terminal_statistics
                        .as_ref()
                        .map(|stats| stats.tokens_generated),
                    execution_time_ms: terminal_statistics
                        .as_ref()
                        .map(|stats| stats.generation_time_ms),
                    audio_duration_secs,
                    rtf: if audio_duration_secs > 0.0 {
                        (generation_time_ms / 1000.0) / audio_duration_secs
                    } else {
                        0.0
                    },
                    first_pcm_ms,
                    last_pcm_ms,
                    total_samples,
                    sample_rate: output_rate,
                    stop_reason: terminal_statistics
                        .as_ref()
                        .and_then(|stats| stats.stop_reason),
                    planned_segment_count,
                    token_unit: if variant.uses_fish_dac_codec() {
                        "semantic_frames"
                    } else {
                        "model_tokens"
                    },
                };
                let _ = send_pump_event(&tx, PumpEvent::Done(Box::new(summary))).await;
            }
            Ok(Err(err)) => {
                let _ = send_pump_event(&tx, fail(core_api_error(err))).await;
            }
            Err(err) => {
                let _ = send_pump_event(
                    &tx,
                    fail(ApiError::internal(format!("Streaming task failed: {err}"))),
                )
                .await;
            }
        }
    }
}

/// HTTP listeners own synchronous OpenAI generation, including while inference
/// has not produced its first chunk. Closing the body must wake this wait.
async fn next_stream_chunk<T>(
    chunks: &mut mpsc::Receiver<AudioChunk>,
    events: &mpsc::Sender<T>,
) -> Option<AudioChunk> {
    tokio::select! {
        biased;
        _ = events.closed() => None,
        chunk = chunks.recv() => chunk,
    }
}

/// The SSE events for one pump event.
fn sse_events(
    event: PumpEvent,
    request_id: &str,
    output: &SpeechOutput,
) -> (
    Option<SpeechStreamEvent>,
    Option<Result<SpeechCompletion, ApiError>>,
) {
    match event {
        PumpEvent::Started { sample_rate } => {
            let mut started = SpeechStreamEvent::new("audio.started", request_id);
            started.sample_rate = Some(sample_rate);
            started.audio_format = Some(output.label());
            started.error = output
                .format
                .fallback
                .as_deref()
                .map(|fallback| format!("Requested format fallback: {fallback}"));
            (Some(started), None)
        }
        PumpEvent::Audio {
            bytes,
            request_id: chunk_request_id,
            sequence,
            sample_count,
            is_final,
            sample_rate,
        } => {
            if bytes.is_empty() {
                return (None, None);
            }
            let mut chunk = SpeechStreamEvent::new("audio.chunk", &chunk_request_id);
            chunk.sequence = (sequence != usize::MAX).then_some(sequence);
            chunk.audio_base64 = Some(base64::engine::general_purpose::STANDARD.encode(bytes));
            chunk.sample_count = Some(sample_count);
            chunk.is_final = Some(is_final);
            chunk.sample_rate = Some(sample_rate);
            chunk.audio_format = Some(output.label());
            (Some(chunk), None)
        }
        PumpEvent::Done(summary) => {
            let mut done = SpeechStreamEvent::new("audio.done", request_id);
            done.timing = Some(summary.timing());
            done.sample_rate = Some(summary.sample_rate);
            done.tokens_generated = summary.tokens_generated;
            done.generation_time_ms = Some(summary.generation_time_ms);
            done.audio_duration_secs = Some(summary.audio_duration_secs);
            done.rtf = Some(summary.rtf);
            done.stop_reason = summary.stop_reason.map(TtsStopReason::as_str);
            (Some(done), Some(Ok(summary.completion())))
        }
        PumpEvent::Failed { error } => {
            let mut event = SpeechStreamEvent::new("audio.failed", request_id);
            event.error = Some(error.message.clone());
            (Some(event), Some(Err(error)))
        }
    }
}

async fn stream_speech_sse(
    state: AppState,
    job: SpeechJob,
    guard: SpeechOutcomeGuard,
) -> Result<Response<Body>, ApiError> {
    let streaming_mode = state.runtime.tts_streaming_mode(job.variant);
    let pump =
        SpeechPump::prepare(&state, &job).map_err(|err| failed(&state, &job.speech_id, err))?;
    let request_id = pump.gen_request.id.clone();
    let output = job.output.clone();
    let speech_id = job.speech_id.clone();
    let (tx, mut rx) = mpsc::channel::<PumpEvent>(stream_event_queue_capacity());
    pump.spawn(tx);

    let outcome_state = state.clone();
    let stream = async_stream::stream! {
        let _guard = guard;
        loop {
            match tokio::time::timeout(Duration::from_secs(10), rx.recv()).await {
                Ok(Some(event)) => {
                    let (sse, ended) = sse_events(event, &request_id, &output);
                    if let Some(sse) = sse {
                        yield Ok::<_, Infallible>(format!("data: {}\n\n", openai_stream_payload(&sse)));
                    }
                    match ended {
                        Some(Ok(completion)) => {
                            outcome_state.speech_outcomes.complete(&speech_id, completion);
                            break;
                        }
                        Some(Err(error)) => {
                            let _ = failed(&outcome_state, &speech_id, error);
                            break;
                        }
                        None => {}
                    }
                }
                Ok(None) => break,
                Err(_) => yield Ok::<_, Infallible>(": keepalive\n\n".to_string()),
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("X-Accel-Buffering", "no")
        // Whether this stream is progressive or one final chunk (TPIPE-3).
        .header("X-Izwi-Streaming-Mode", streaming_mode.as_str())
        .header("X-Izwi-Speech-Id", job.speech_id.as_str())
        .header("X-SGLang-Omni-Speech-Id", job.speech_id.as_str())
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        )
        .body(Body::from_stream(stream))
        .unwrap())
}

/// `stream_format: "audio"`: the encoded file as a chunked body while it is
/// generated. Headers wait for the first audio, so the rate is known and a
/// failure before any audio is an ordinary error response; a failure after
/// it ends the body early, and the outcome lookup says why.
async fn stream_speech_audio(
    state: AppState,
    job: SpeechJob,
    guard: SpeechOutcomeGuard,
) -> Result<Response<Body>, ApiError> {
    let streaming_mode = state.runtime.tts_streaming_mode(job.variant);
    let pump =
        SpeechPump::prepare(&state, &job).map_err(|err| failed(&state, &job.speech_id, err))?;
    let speech_id = job.speech_id.clone();
    let (tx, mut rx) = mpsc::channel::<PumpEvent>(stream_event_queue_capacity());
    pump.spawn(tx);

    let sample_rate = match rx.recv().await {
        Some(PumpEvent::Started { sample_rate }) => sample_rate,
        Some(PumpEvent::Failed { error }) => return Err(failed(&state, &speech_id, error)),
        _ => {
            return Err(failed(
                &state,
                &speech_id,
                ApiError::internal("The speech stream ended before any audio"),
            ))
        }
    };

    let outcome_state = state.clone();
    let body_speech_id = speech_id.clone();
    let body = async_stream::stream! {
        let _guard = guard;
        while let Some(event) = rx.recv().await {
            match event {
                PumpEvent::Audio { bytes, .. } if !bytes.is_empty() => {
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from(bytes));
                }
                PumpEvent::Done(summary) => {
                    outcome_state
                        .speech_outcomes
                        .complete(&body_speech_id, summary.completion());
                    break;
                }
                PumpEvent::Failed { error } => {
                    let message = error.message.clone();
                    let _ = failed(&outcome_state, &body_speech_id, error);
                    yield Err(std::io::Error::other(message));
                    break;
                }
                PumpEvent::Audio { .. } | PumpEvent::Started { .. } => {}
            }
        }
    };

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, job.output.content_type())
        .header(header::CACHE_CONTROL, "no-cache, no-transform")
        .header("X-Accel-Buffering", "no")
        .header("X-Audio-Sample-Rate", sample_rate.to_string())
        .header("X-Audio-Channels", "1")
        .header("X-Izwi-Streaming-Mode", streaming_mode.as_str())
        .header("X-Izwi-Speech-Id", speech_id.as_str())
        .header("X-SGLang-Omni-Speech-Id", speech_id.as_str())
        .header("X-Actual-Response-Format", job.output.label())
        .header(
            "Access-Control-Expose-Headers",
            SPEECH_RESPONSE_EXPOSED_HEADERS,
        );
    if let Some(fallback) = job.output.format.fallback.as_deref() {
        builder = builder.header("X-Response-Format-Fallback", fallback);
    }
    builder
        .body(Body::from_stream(body))
        .map_err(|e| ApiError::internal(e.to_string()))
}

/// `GET /v1/audio/speech/{id}`: the outcome of a speech request this process
/// served, while it is remembered (`IZWI_SPEECH_OUTCOME_CAPACITY`,
/// `IZWI_SPEECH_OUTCOME_TTL_SECS`).
pub async fn get_speech(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    Path(speech_id): Path<String>,
) -> Result<Json<SpeechOutcome>, ApiError> {
    state
        .speech_outcomes
        .get(&speech_id, ctx.tenant_key())
        .map(Json)
        .ok_or_else(|| {
            ApiError::not_found(format!(
                "No speech request with id {speech_id} (outcomes are kept in memory for a limited time)"
            ))
        })
}

pub(crate) fn build_generation_request(
    req: &SpeechRequest,
    correlation_id: String,
    streaming: bool,
    variant: ModelVariant,
) -> GenerationRequest {
    let mut gen_config = GenerationConfig {
        streaming,
        ..GenerationConfig::default()
    };
    if let Some(temp) = req.temperature {
        gen_config.options.temperature = temp;
    }
    if let Some(speed) = req.speed {
        gen_config.options.speed = speed;
    }
    if let Some(max_tokens) = resolve_tts_output_frames(
        variant,
        &req.input,
        req.max_output_tokens.or(req.max_tokens),
    ) {
        gen_config.options.max_tokens = if variant.plans_own_speech_segments() {
            req.max_output_tokens.or(req.max_tokens).unwrap_or(0)
        } else {
            max_tokens
        };
    }
    if let Some(top_k) = req.top_k {
        gen_config.options.top_k = top_k;
    }
    gen_config.options.speech = req.sampling.options(
        req.temperature,
        crate::api::tts_policy::speech_top_k(variant, req.top_k),
    );
    gen_config.options.speaker = req
        .voice
        .as_ref()
        .and_then(SpeechVoice::name)
        .map(str::to_string);

    GenerationRequest {
        id: uuid::Uuid::new_v4().to_string(),
        model_variant: Some(variant),
        correlation_id: Some(correlation_id),
        runtime_context: Default::default(),
        text: req.input.clone(),
        config: gen_config,
        language: req.language.clone(),
        reference_audio: req.reference_audio.clone(),
        reference_text: req.reference_text.clone(),
        reference_language: req.reference_language.clone(),
        voice_description: req.instructions.clone(),
        speaker_references: crate::api::speech_dialogue::speaker_references(&req.references),
        chunk_history: Vec::new(),
    }
}

fn normalize_optional_trimmed(raw: Option<String>) -> Option<String> {
    let trimmed = raw.unwrap_or_default().trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn has_non_empty_text(raw: &str) -> bool {
    !raw.trim().is_empty()
}

/// The formats a request may name, for error messages.
fn supported_format_names() -> String {
    let mut names = vec!["pcm"];
    names.extend(
        AudioFormat::available()
            .into_iter()
            .map(AudioFormat::as_str),
    );
    names.join(", ")
}

/// The format `format` names. OpenAI's `pcm` is 24 kHz PCM16 (`audio/pcm`);
/// Izwi's `pcm_i16`/`pcm16`/`raw_i16` keep the model's rate. A recognized
/// format this build cannot encode is a 400, or WAV when the request allows
/// a fallback.
pub(crate) fn parse_response_format(
    format: &str,
    allow_fallback: bool,
) -> Result<ResolvedSpeechFormat, ApiError> {
    let normalized = format.trim().to_ascii_lowercase();
    if normalized == "pcm" {
        return Ok(ResolvedSpeechFormat {
            format: AudioFormat::PcmI16,
            label: "pcm",
            content_type: "audio/pcm",
            forced_rate: Some(OPENAI_PCM_RATE),
            fallback: None,
        });
    }
    let Some(format) = AudioFormat::parse(&normalized) else {
        return Err(ApiError::invalid_param(
            "response_format",
            format!(
                "Unsupported response_format: {normalized}. Supported: {}",
                supported_format_names()
            ),
        ));
    };
    if format.is_available() {
        return Ok(ResolvedSpeechFormat {
            format,
            label: format.as_str(),
            content_type: format.content_type(),
            forced_rate: None,
            fallback: None,
        });
    }
    if allow_fallback {
        return Ok(ResolvedSpeechFormat {
            format: AudioFormat::Wav,
            label: "wav",
            content_type: AudioFormat::Wav.content_type(),
            forced_rate: None,
            fallback: Some(format!("{normalized}->wav")),
        });
    }
    let reason = EncoderOptions::new(format, 24_000)
        .validate()
        .err()
        .map(|err| err.to_string())
        .unwrap_or_else(|| format!("{format} encoding is not available"));
    Err(ApiError::invalid_param(
        "response_format",
        format!(
            "{reason} Supported: {}. Set `allow_format_fallback: true` to receive WAV bytes instead.",
            supported_format_names()
        ),
    ))
}

/// How the request asked to be delivered: OpenAI's `stream_format`
/// (`audio` or `sse`), or Izwi's `stream: true` (SSE).
fn resolve_delivery(req: &SpeechRequest) -> Result<SpeechDelivery, ApiError> {
    let stream_bool = req.stream.unwrap_or(false);
    let stream_format = req
        .stream_format
        .as_deref()
        .map(|value| value.trim().to_ascii_lowercase());
    match stream_format.as_deref() {
        None | Some("") if stream_bool => Ok(SpeechDelivery::Sse),
        None | Some("") => Ok(SpeechDelivery::Buffered),
        Some("sse") => Ok(SpeechDelivery::Sse),
        Some("audio") => Ok(SpeechDelivery::AudioStream),
        Some(other) => Err(ApiError::invalid_param(
            "stream_format",
            format!("Unsupported stream_format: {other}. Supported values: audio, sse"),
        )),
    }
}

/// Non-final statistics are deltas; terminal statistics are cumulative. Audio
/// samples never imply a token count, because codec geometry varies by model.
fn accumulate_stream_statistics(total: &mut Option<izwi_core::ChunkStats>, chunk: &AudioChunk) {
    let Some(stats) = chunk.stats.as_ref() else {
        return;
    };
    match total.as_mut() {
        Some(total) if !chunk.is_final => {
            total.tokens_generated = total
                .tokens_generated
                .saturating_add(stats.tokens_generated);
            total.generation_time_ms += stats.generation_time_ms;
        }
        _ => *total = Some(stats.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[tokio::test]
    async fn disconnected_openai_listener_wakes_before_first_pcm() {
        let (_producer, mut chunks) = mpsc::channel(2);
        let (events, listener) = mpsc::channel::<PumpEvent>(2);
        drop(listener);
        assert!(tokio::time::timeout(
            Duration::from_millis(100),
            next_stream_chunk(&mut chunks, &events)
        )
        .await
        .expect("disconnect must not await first PCM")
        .is_none());
    }

    /// TPIPE-24: unset sampling stays unset for the engine (so family
    /// defaults apply), and the izwi extension fields reach it.
    #[test]
    fn speech_sampling_fields_reach_the_engine_and_unset_stays_unset() {
        let unset: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "Qwen3-TTS-12Hz-0.6B-CustomVoice", "input": "hello", "voice": "ryan"
        }))
        .unwrap();
        let generation = build_generation_request(
            &unset,
            "c".into(),
            false,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        );
        assert_eq!(
            generation.config.options.speech,
            izwi_core::SpeechSamplingOptions::default()
        );

        let set: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "VibeVoice-1.5B", "input": "hello", "temperature": 0.5, "top_k": 12,
            "top_p": 0.9, "seed": 7, "subtalker_temperature": 0.6, "subtalker_top_k": 20,
            "subtalker_top_p": 0.95, "subtalker_do_sample": false, "cfg_scale": 1.5,
            "diffusion_steps": 20, "repetition_penalty": 1.2
        }))
        .unwrap();
        let speech =
            build_generation_request(&set, "c".into(), false, ModelVariant::VibeVoice15BTts)
                .config
                .options
                .speech;
        assert_eq!(
            (speech.temperature, speech.top_k, speech.top_p, speech.seed),
            (Some(0.5), Some(12), Some(0.9), Some(7))
        );
        assert_eq!(
            (
                speech.subtalker_temperature,
                speech.subtalker_top_k,
                speech.subtalker_top_p,
                speech.subtalker_do_sample
            ),
            (Some(0.6), Some(20), Some(0.95), Some(false))
        );
        assert_eq!(
            (
                speech.cfg_scale,
                speech.diffusion_steps,
                speech.repetition_penalty
            ),
            (Some(1.5), Some(20), Some(1.2))
        );
    }

    #[test]
    fn voxtral_tts_omitted_max_tokens_uses_text_sized_auto_budget() {
        let req = SpeechRequest {
            model: "Voxtral-4B-TTS-2603".to_string(),
            input: "The costs split cleanly into three buckets".to_string(),
            voice: Some("casual_male".into()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::Voxtral4BTts2603,
        );
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::Voxtral4BTts2603, &req.input, None)
                .expect("Voxtral frame hint")
        );
    }

    #[test]
    fn vibevoice_tts_omitted_max_tokens_uses_text_sized_auto_budget() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "The costs split cleanly into three buckets".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: Some("hello".to_string()),
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::VibeVoice15BTts, &req.input, None)
                .expect("VibeVoice frame hint")
        );
    }

    #[test]
    fn fish_s2_tts_preserves_automatic_intent_until_segment_planning() {
        let req = SpeechRequest {
            model: "FishAudio-S2-Pro".to_string(),
            input: "The costs split cleanly into three buckets".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: Some("hello".to_string()),
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::FishAudioS2Pro,
        );
        assert_eq!(
            generation.config.options.max_tokens, 0,
            "automatic Fish requests must be planned before per-segment budgeting"
        );
    }

    #[test]
    fn qwen_tts_omitted_max_tokens_uses_shared_text_sized_budget() {
        let req = SpeechRequest {
            model: "Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string(),
            input: "The quick brown fox jumps over the lazy dog.".to_string(),
            voice: Some("Aiden".into()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        );
        assert_eq!(
            generation.config.options.max_tokens,
            resolve_tts_output_frames(ModelVariant::Qwen3Tts12Hz06BCustomVoice, &req.input, None,)
                .expect("Qwen frame hint")
        );
    }

    #[test]
    fn vibevoice_speech_requests_require_reference_voice() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let err = validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect_err("missing reference should fail");
        assert!(err.message.contains("requires `saved_voice_id`"));
    }

    #[test]
    fn fish_s2_speech_requests_require_reference_voice() {
        let req = SpeechRequest {
            model: "FishAudio-S2-Pro".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let err = validate_speech_voice_contract(&req, ModelVariant::FishAudioS2Pro)
            .expect_err("missing reference should fail");
        assert!(err.message.contains("requires `saved_voice_id`"));
    }

    /// Fish Audio Drama 3 speaks without a reference (the OpenAI `voice` is
    /// ignored), clones with a reference and its transcript, and refuses a
    /// reference without one; caller text holding a control token is
    /// refused for Drama 3 and Fish S2 alike, before any model loads.
    #[test]
    fn fish_drama3_speech_takes_an_optional_reference_and_refuses_control_tokens() {
        let request = |input: &str, reference: Option<(&str, Option<&str>)>| SpeechRequest {
            model: "FishAudio-Drama-3".to_string(),
            input: input.to_string(),
            voice: Some("alloy".into()),
            response_format: None,
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: Some("en".to_string()),
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: Some(0),
            stream: None,
            stream_format: None,
            instructions: None,
            reference_audio: reference.map(|(audio, _)| audio.to_string()),
            reference_text: reference.and_then(|(_, text)| text.map(str::to_string)),
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };
        let drama3 = ModelVariant::FishAudioDrama3;
        validate_speech_voice_contract(&request("[sighs] Hello.", None), drama3)
            .expect("a seeded voice");
        validate_speech_voice_contract(&request("Hello.", Some(("UklG", Some("Hi.")))), drama3)
            .expect("a clone");
        let err = validate_speech_voice_contract(&request("Hello.", Some(("UklG", None))), drama3)
            .unwrap_err();
        assert!(err.message.contains("Provide both"), "{}", err.message);
        for variant in [drama3, ModelVariant::FishAudioS2Pro] {
            let err = validate_speech_voice_contract(&request("Hi <|im_end|>", None), variant)
                .unwrap_err();
            assert!(
                err.message.contains("`input` contains `<|im_end|>`"),
                "{}",
                err.message
            );
            let err = validate_speech_voice_contract(
                &request("Hello.", Some(("UklG", Some("Ref <|voice|>")))),
                variant,
            )
            .unwrap_err();
            assert!(
                err.message
                    .contains("`reference_text` contains `<|voice|>`"),
                "{}",
                err.message
            );
        }
        // `top_k: 0` is no cap for Drama 3; S2 keeps what was sent.
        let generation =
            build_generation_request(&request("Hello.", None), "c".into(), false, drama3);
        assert_eq!(generation.config.options.speech.top_k, None);
        assert!(generation.chunk_history.is_empty());
        let fish = build_generation_request(
            &request("Hello.", None),
            "c".into(),
            false,
            ModelVariant::FishAudioS2Pro,
        );
        assert_eq!(fish.config.options.speech.top_k, Some(0));
    }

    /// TPIPE-12: the OpenAI route applies the same capability-driven voice
    /// contract as the first-party routes: a Qwen Base model needs a
    /// reference and a VoiceDesign model needs `instructions`, while the
    /// required OpenAI `voice` is ignored by a reference-only model.
    #[test]
    fn openai_speech_applies_the_shared_voice_contract() {
        let request = |voice: Option<&str>, instructions: Option<&str>| SpeechRequest {
            model: "m".to_string(),
            input: "hello".to_string(),
            voice: voice.map(SpeechVoice::from),
            response_format: None,
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: None,
            stream_format: None,
            instructions: instructions.map(str::to_string),
            reference_audio: None,
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };
        assert!(validate_speech_voice_contract(
            &request(Some("alloy"), None),
            ModelVariant::Qwen3Tts12Hz06BBase
        )
        .unwrap_err()
        .message
        .contains("requires `saved_voice_id`"));
        assert!(validate_speech_voice_contract(
            &request(None, None),
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign
        )
        .unwrap_err()
        .message
        .contains("requires a voice description"));
        assert!(validate_speech_voice_contract(
            &request(None, Some("A calm narrator")),
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign
        )
        .is_ok());
        assert!(validate_speech_voice_contract(
            &request(Some("af_heart"), None),
            ModelVariant::Kokoro82M
        )
        .is_ok());
    }

    /// TPIPE-11: the SSE events also carry OpenAI's `speech.audio.delta` and
    /// `speech.audio.done` shape.
    #[test]
    fn sse_events_carry_the_openai_speech_stream_shape() {
        let event = |name: &'static str| SpeechStreamEvent {
            event: name,
            timing: None,
            request_id: None,
            sequence: Some(0),
            audio_base64: Some("AAEC".into()),
            sample_count: None,
            is_final: None,
            sample_rate: None,
            audio_format: None,
            tokens_generated: Some(12),
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
            stop_reason: None,
        };
        let delta: serde_json::Value =
            serde_json::from_str(&openai_stream_payload(&event("audio.chunk"))).unwrap();
        assert_eq!(delta["type"], "speech.audio.delta");
        assert_eq!(delta["audio"], "AAEC");
        assert_eq!(delta["event"], "audio.chunk");
        assert_eq!(delta["audio_base64"], "AAEC");
        let done: serde_json::Value =
            serde_json::from_str(&openai_stream_payload(&event("audio.done"))).unwrap();
        assert_eq!(done["type"], "speech.audio.done");
        assert_eq!(done["usage"]["output_tokens"], 12);
        assert_eq!(done["usage"]["input_tokens"], 0);
        assert_eq!(done["usage"]["total_tokens"], 12);
        let failed: serde_json::Value =
            serde_json::from_str(&openai_stream_payload(&event("audio.failed"))).unwrap();
        assert_eq!(failed["type"], "error");
    }

    #[test]
    fn direct_reference_audio_and_text_must_be_paired() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: None,
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: None,
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        let err = validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect_err("partial reference should fail");
        assert!(err.message.contains("Provide both `reference_audio`"));
    }

    #[test]
    fn generation_request_preserves_vibevoice_reference_fields_and_voice_label() {
        let req = SpeechRequest {
            model: "VibeVoice-1.5B".to_string(),
            input: "hello".to_string(),
            voice: Some("Speaker 1".into()),
            response_format: Some("wav".to_string()),
            allow_format_fallback: None,
            sample_rate: None,
            bitrate: None,
            speed: None,
            language: None,
            temperature: None,
            max_tokens: None,
            max_output_tokens: None,
            top_k: None,
            stream: Some(false),
            stream_format: None,
            instructions: None,
            reference_audio: Some("UklGRg==".to_string()),
            reference_text: Some("reference words".to_string()),
            reference_language: None,
            saved_voice_id: None,
            references: Vec::new(),
            script: None,
            sampling: Default::default(),
        };

        validate_speech_voice_contract(&req, ModelVariant::VibeVoice15BTts)
            .expect("valid reference");
        let generation = build_generation_request(
            &req,
            "test-correlation".to_string(),
            false,
            ModelVariant::VibeVoice15BTts,
        );

        assert_eq!(
            generation.config.options.speaker.as_deref(),
            Some("Speaker 1")
        );
        assert_eq!(generation.reference_audio.as_deref(), Some("UklGRg=="));
        assert_eq!(
            generation.reference_text.as_deref(),
            Some("reference words")
        );
    }

    #[test]
    fn compressed_formats_resolve_to_their_codecs() {
        for (name, format, content_type) in [
            ("mp3", AudioFormat::Mp3, "audio/mpeg"),
            ("opus", AudioFormat::Opus, "audio/ogg"),
            ("ogg", AudioFormat::Opus, "audio/ogg"),
            ("flac", AudioFormat::Flac, "audio/flac"),
            ("wav", AudioFormat::Wav, "audio/wav"),
            ("pcm_i16", AudioFormat::PcmI16, "application/octet-stream"),
        ] {
            let resolved = parse_response_format(name, false).expect(name);
            assert_eq!(resolved.format, format, "{name}");
            assert_eq!(resolved.content_type, content_type, "{name}");
            assert_eq!(resolved.forced_rate, None, "{name}");
            assert!(resolved.fallback.is_none());
        }
        // OpenAI's pcm is 24 kHz whatever the model's rate.
        let pcm = parse_response_format("pcm", false).unwrap();
        assert_eq!(
            (pcm.format, pcm.label, pcm.content_type, pcm.forced_rate),
            (AudioFormat::PcmI16, "pcm", "audio/pcm", Some(24_000))
        );
        let unknown = parse_response_format("m4a", false).unwrap_err();
        assert_eq!(unknown.status, StatusCode::BAD_REQUEST);
        assert!(unknown.message.contains("mp3"), "{}", unknown.message);
    }

    #[test]
    fn aac_needs_an_explicit_fallback_and_says_why() {
        let error = parse_response_format("aac", false).expect_err("aac is not built in");
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("patent"), "{}", error.message);
        assert!(error.message.contains("allow_format_fallback"));

        let resolved = parse_response_format("aac", true).expect("fallback");
        assert_eq!(resolved.format, AudioFormat::Wav);
        assert_eq!(resolved.label, "wav");
        assert_eq!(resolved.fallback.as_deref(), Some("aac->wav"));
    }

    #[test]
    fn output_options_check_rates_and_bitrates_before_any_work() {
        let output = |format: &str, rate: Option<u32>, bitrate: Option<u32>| {
            SpeechOutput::resolve(Some(format), false, rate, bitrate)
        };
        assert!(output("mp3", None, Some(64_000)).is_ok());
        assert!(output("opus", Some(48_000), Some(64_000)).is_ok());
        assert!(
            output("opus", Some(44_100), None).is_err(),
            "not an Opus rate"
        );
        assert!(
            output("wav", None, Some(64_000)).is_err(),
            "wav takes no bitrate"
        );
        assert!(output("mp3", None, Some(1_000)).is_err());
        let pcm = output("pcm", None, None).unwrap();
        assert_eq!(pcm.output_rate(44_100), 24_000);
        assert_eq!(
            output("pcm_i16", None, None).unwrap().output_rate(44_100),
            44_100
        );
    }

    #[test]
    fn delivery_follows_stream_format_then_stream() {
        let delivery = |stream: Option<bool>, stream_format: Option<&str>| {
            resolve_delivery(&SpeechRequest {
                model: "Kokoro-82M".to_string(),
                input: "hello".to_string(),
                stream,
                stream_format: stream_format.map(str::to_string),
                ..Default::default()
            })
        };
        assert_eq!(delivery(None, None).unwrap(), SpeechDelivery::Buffered);
        assert_eq!(delivery(Some(true), None).unwrap(), SpeechDelivery::Sse);
        assert_eq!(
            delivery(Some(false), Some("sse")).unwrap(),
            SpeechDelivery::Sse
        );
        assert_eq!(
            delivery(None, Some("audio")).unwrap(),
            SpeechDelivery::AudioStream
        );
        assert_eq!(
            delivery(Some(true), Some("audio")).unwrap(),
            SpeechDelivery::AudioStream
        );
        assert!(delivery(None, Some("chunked")).is_err());
    }

    #[test]
    fn voice_objects_name_saved_voices() {
        let req: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "Qwen3-TTS-12Hz-0.6B-Base", "input": "hi", "voice": {"id": " voice_abc "}
        }))
        .unwrap();
        let req = normalize_speech_request(req).unwrap();
        assert_eq!(req.saved_voice_id.as_deref(), Some("voice_abc"));
        assert_eq!(req.voice.as_ref().and_then(SpeechVoice::name), None);

        let clash: SpeechRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "input": "hi", "voice": {"id": "a"}, "saved_voice_id": "b"
        }))
        .unwrap();
        assert!(normalize_speech_request(clash).is_err());

        let named: SpeechRequest = serde_json::from_value(
            serde_json::json!({"model": "m", "input": "hi", "voice": "alloy"}),
        )
        .unwrap();
        assert_eq!(
            named.voice.as_ref().and_then(SpeechVoice::name),
            Some("alloy")
        );
        assert!(check_instructions_length(Some(&"x".repeat(4_097))).is_err());
        assert!(check_instructions_length(Some(&"x".repeat(4_096))).is_ok());
    }

    #[test]
    fn sse_started_waits_for_pcm_and_reports_output_rate_format_and_fallback() {
        let output = SpeechOutput::resolve(Some("aac"), true, None, None).unwrap();
        let (event, ended) = sse_events(
            PumpEvent::Started {
                sample_rate: 44_100,
            },
            "req",
            &output,
        );
        let event = event.unwrap();
        assert!(ended.is_none());
        assert_eq!(event.event, "audio.started");
        assert_eq!(event.sample_rate, Some(44_100));
        assert_eq!(event.audio_format, Some("wav"));
        assert_eq!(
            event.error.as_deref(),
            Some("Requested format fallback: aac->wav")
        );
        // A pump event carrying no bytes (a codec holding a partial frame)
        // is not an SSE event.
        let (empty, _) = sse_events(
            PumpEvent::Audio {
                bytes: Vec::new(),
                request_id: "req".into(),
                sequence: 0,
                sample_count: 10,
                is_final: false,
                sample_rate: 24_000,
            },
            "req",
            &output,
        );
        assert!(empty.is_none());
    }

    #[test]
    fn streaming_terminal_only_statistics_replace_deltas_and_unknown_stays_unknown() {
        let mut statistics = None;
        let mut first = AudioChunk::new("fish".into(), 0, vec![0.0; 2048]);
        accumulate_stream_statistics(&mut statistics, &first);
        assert!(statistics.is_none());
        first.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 10.0,
            tokens_generated: 1,
            rtf: 0.2,
            stop_reason: None,
        });
        accumulate_stream_statistics(&mut statistics, &first);
        let mut terminal = AudioChunk::final_chunk("fish".into(), 1, Vec::new());
        terminal.stats = Some(izwi_core::ChunkStats {
            generation_time_ms: 25.0,
            tokens_generated: 2,
            rtf: 0.3,
            stop_reason: None,
        });
        accumulate_stream_statistics(&mut statistics, &terminal);
        let statistics = statistics.unwrap();
        assert_eq!(statistics.tokens_generated, 2);
        assert_eq!(statistics.generation_time_ms, 25.0);
    }

    /// D7: a streamed WAV is one RIFF header (sizes "unknown") followed by
    /// PCM, so the concatenated deltas are one playable file, as with every
    /// other format.
    #[test]
    fn streamed_wav_deltas_concatenate_into_one_file() {
        let output = SpeechOutput::resolve(Some("wav"), false, None, None).unwrap();
        let mut encoder = output.stream_encoder(44_100).unwrap();
        let mut bytes = Vec::new();
        for chunk in [&[0.0f32, 0.25][..], &[-0.25f32][..]] {
            bytes.extend(encoder.push(chunk).unwrap());
        }
        bytes.extend(encoder.finish().unwrap());
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(
            bytes.windows(4).filter(|window| *window == b"RIFF").count(),
            1
        );
        assert_eq!(bytes.len(), 44 + 3 * 2);
        assert_eq!(
            &bytes[44..],
            &izwi_audio_codec::pcm16_le(&[0.0, 0.25, -0.25])[..]
        );
    }

    #[test]
    fn buffered_wav_uses_the_generated_sample_rate() {
        let output = SpeechOutput::resolve(Some("wav"), false, None, None).unwrap();
        let bytes = output
            .encode_file(&[0.0, 0.25, -0.25], 16_000)
            .expect("encode wav");
        let reader = hound::WavReader::new(Cursor::new(bytes)).expect("decode wav");

        assert_eq!(reader.spec().sample_rate, 16_000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.duration(), 3);
    }

    #[test]
    fn speech_stream_chunk_event_includes_audio_rate_and_format() {
        let event = SpeechStreamEvent {
            event: "audio.chunk",
            timing: None,
            request_id: Some("req".to_string()),
            sequence: Some(3),
            audio_base64: Some("AA==".to_string()),
            sample_count: Some(1),
            is_final: Some(false),
            sample_rate: Some(48_000),
            audio_format: Some("pcm_i16"),
            tokens_generated: None,
            generation_time_ms: None,
            audio_duration_secs: None,
            rtf: None,
            error: None,
            stop_reason: None,
        };

        let value = serde_json::to_value(event).expect("serialize event");

        assert_eq!(value["sample_rate"], 48_000);
        assert_eq!(value["audio_format"], "pcm_i16");
    }
}
