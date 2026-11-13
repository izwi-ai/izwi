//! The `/v1/realtime` session actor: one per socket. It owns the input
//! buffer and its VAD, the conversation, and the responses in flight, and
//! answers each client event in order.
//!
//! Input runs through the voice socket's `StreamingInputState` (Earshot VAD,
//! endpointing, pre-roll, the speech-evidence gate). `speech_started` is
//! sent once speech is confirmed (Izwi's barge-in threshold), dated back to
//! where it began, so a cough or speaker echo never stops a client's
//! playback (D4); with `interrupt_response` it then cancels the response in
//! progress, which Pipecat relies on.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures::{Sink, SinkExt, Stream, StreamExt};
use izwi_core::SessionModelLeases;
use izwi_vad::{configured_snr_margin_db, sanitize_snr_margin};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tracing::{debug, warn};

use super::audio::{base64, decode_input, pcm16_bytes, wav};
use super::events::{parse_client_event, ClientEvent, Ids, RealtimeError};
use super::items::{ContentPart, Item, ItemAudio, ItemKind, ItemStatus, ItemStore, Position, Role};
use super::live::{transcribes_live, LiveTranscription, LiveTurn};
use super::persist::History;
use super::profiles::{resolve_profile, speaker_for, transcription_model, RealtimeProfile};
use super::respond::{
    lock_items, mcp_configs, mcp_tools_for, run_response, Outbound, PendingTranscript,
    PendingTranscripts, ResponseContext, ResponsePlan, ResponseSummary, SharedMcp,
};
use super::session_config::{
    parse_max_output_tokens, parse_metadata, parse_tools, AudioFormat, Modality, SessionConfig,
    Voice,
};
use super::transcribe::Transcription;
use super::voices::{resolve_custom_voice, resolve_named_voice};
use crate::app::mcp::McpServerConfig;
use crate::app::realtime_transport::{spawn_writer, OutboundLimits, OutboundMailbox};
use crate::app::voice_realtime::{
    normalize_stream_vad_threshold, StreamingInputConfig, StreamingInputState, TtsVoice,
    TurnControl, UtteranceOutcome, DEFAULT_BARGE_IN_MIN_SPEECH_MS, DEFAULT_STREAM_MIN_SPEECH_MS,
    MAX_UTTERANCE_PCM16_SAMPLES, SESSION_ACTOR_TICK,
};
use crate::state::AppState;
use crate::voice_defaults::DEFAULT_VOICE_AGENT_SYSTEM_PROMPT;

/// The shortest buffer a manual commit takes (OpenAI's rule).
const MIN_COMMIT_MS: f64 = 100.0;
/// The longest utterance server VAD lets run before it commits (D17).
const MAX_UTTERANCE_MS: u32 = 60_000;
/// Events an OpenAI session may queue: many small deltas per response.
const OUTBOUND_ITEMS: usize = 4_096;
/// How long a session lives, as OpenAI's do (sent as `expires_at`).
const SESSION_LIFETIME: Duration = Duration::from_secs(60 * 60);

/// What a session is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionKind {
    /// A conversation: input, responses.
    Realtime,
    /// `?intent=transcription`: input and its transcripts, no responses.
    Transcription,
}

struct ActiveResponse {
    id: String,
    control: TurnControl,
    task: tokio::task::JoinHandle<()>,
    generation: Option<u64>,
    default_conversation: bool,
}

enum Input {
    /// Server or semantic VAD cuts turns.
    Vad(Box<StreamingInputState>),
    /// `turn_detection: null`: the client commits.
    Manual(Vec<i16>),
}

/// Speech the VAD heard start, and the user item it will become.
struct Speech {
    item_id: String,
    start_ms: u64,
    announced: bool,
}

struct Session {
    state: AppState,
    correlation_id: String,
    kind: SessionKind,
    out: Outbound,
    profile: RealtimeProfile,
    config: SessionConfig,
    conversation_id: String,
    items: Arc<Mutex<ItemStore>>,
    input: Input,
    frame_seq: u32,
    /// Milliseconds of input appended in the session: OpenAI's
    /// `audio_start_ms`/`audio_end_ms` clock.
    input_ms: f64,
    speech: Option<Speech>,
    /// The turn in progress, when its model transcribes it as it is spoken.
    live: Option<LiveTurn>,
    responses: Vec<ActiveResponse>,
    next_generation: u64,
    /// A VAD turn ended while a response was still running; answer it next.
    pending_auto_response: bool,
    /// The session has spoken: OpenAI fixes the voice from then on.
    voice_locked: bool,
    /// A saved voice the session speaks in, resolved at `session.update`.
    custom_voice: Option<TtsVoice>,
    model_leases: SessionModelLeases,
    pending_transcripts: PendingTranscripts,
    lane: Arc<tokio::sync::Mutex<()>>,
    /// The session's connections to its MCP servers, and the servers whose
    /// tools were last announced.
    mcp: SharedMcp,
    mcp_announced: Vec<McpServerConfig>,
    finished_tx: mpsc::UnboundedSender<(String, ResponseSummary)>,
    /// The voice-store record of the session, when `IZWI_REALTIME_PERSIST`
    /// is on (D5).
    history: Option<Arc<History>>,
    /// When the conversation last moved, for `idle_timeout_ms`.
    last_activity: Instant,
    idle_fired: bool,
}

fn input_for(config: &SessionConfig) -> Input {
    let Some(turn_detection) = config.turn_detection.as_ref() else {
        return Input::Manual(Vec::new());
    };
    let (threshold, prefix_padding_ms, silence_ms) = turn_detection.vad();
    Input::Vad(Box::new(StreamingInputState::new(StreamingInputConfig {
        vad_threshold: normalize_stream_vad_threshold(Some(threshold)),
        min_speech_ms: DEFAULT_STREAM_MIN_SPEECH_MS,
        silence_duration_ms: silence_ms,
        max_utterance_ms: MAX_UTTERANCE_MS,
        pre_roll_ms: prefix_padding_ms,
        barge_in_min_speech_ms: DEFAULT_BARGE_IN_MIN_SPEECH_MS,
        snr_margin_db: sanitize_snr_margin(configured_snr_margin_db()),
        input_sample_rate_hint: Some(config.input_format.sample_rate()),
        adaptive_silence_floor_ms: None,
        early_transcription: false,
    })))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Cut `text` to the share of it `heard_ms` of `audio_ms` covers, at a word.
fn heard_share(text: &str, heard_ms: u64, audio_ms: u64) -> String {
    if audio_ms == 0 || heard_ms >= audio_ms {
        return text.to_string();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let keep = (words.len() as u64 * heard_ms / audio_ms) as usize;
    words[..keep.min(words.len())].join(" ")
}

/// What a session runs, resolved when its socket is opened.
pub(crate) struct SessionSpec {
    pub(crate) profile: RealtimeProfile,
    pub(crate) requested_model: String,
    pub(crate) kind: SessionKind,
    /// A session config applied before the session starts: an ephemeral
    /// key's, or a media-stream call's.
    pub(crate) bound_session: Option<Value>,
}

impl SessionSpec {
    /// The session `model` and `intent` ask for; either may come from the
    /// session config an ephemeral key binds instead.
    pub(crate) fn resolve(
        model: Option<&str>,
        intent: Option<&str>,
        bound_session: Option<Value>,
    ) -> Result<Self, String> {
        let bound = |field: &str| {
            bound_session
                .as_ref()
                .and_then(|session| session.get(field))
                .and_then(Value::as_str)
        };
        let kind = match intent.map(str::trim).or(bound("type")) {
            None | Some("" | "realtime" | "conversation") => SessionKind::Realtime,
            Some("transcription") => SessionKind::Transcription,
            Some(other) => {
                return Err(format!(
                    "Unsupported intent `{other}`; use `realtime` or `transcription`."
                ))
            }
        };
        let model = model.or(bound("model"));
        let profile = resolve_profile(model)?;
        let requested_model = model
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .unwrap_or(&profile.id)
            .to_string();
        Ok(Self {
            profile,
            requested_model,
            kind,
            bound_session,
        })
    }
}

/// Run one `/v1/realtime` socket to its end.
pub(crate) async fn handle_socket(
    socket: WebSocket,
    state: AppState,
    correlation_id: String,
    spec: SessionSpec,
) {
    let (ws_tx, ws_rx) = socket.split();
    run_session(ws_tx, ws_rx, state, correlation_id, spec).await;
}

/// Run one session to its end over a socket's two halves. They carry
/// `/v1/realtime` events: the socket's own, or another framing translated
/// to them (`app::media_stream`).
pub(crate) async fn run_session<Tx, Rx, E>(
    mut ws_tx: Tx,
    mut ws_rx: Rx,
    state: AppState,
    correlation_id: String,
    spec: SessionSpec,
) where
    Tx: Sink<Message> + Unpin + Send + 'static,
    Rx: Stream<Item = Result<Message, E>> + Unpin,
{
    let SessionSpec {
        profile,
        requested_model,
        kind,
        bound_session,
    } = spec;
    let ids = Arc::new(Ids::new());
    let Some(_session_permit) = state.try_acquire_realtime_session() else {
        let error = RealtimeError::server("Realtime session capacity is exhausted; retry shortly.");
        let mut event = error.to_event(&ids);
        event["error"]["code"] = json!("session_capacity");
        let _ = ws_tx.send(Message::Text(event.to_string().into())).await;
        // 1013 (try again later) is a close openai-python retries.
        let _ = ws_tx
            .send(Message::Close(Some(CloseFrame {
                code: 1013,
                reason: "realtime_session_capacity".into(),
            })))
            .await;
        return;
    };

    let tuning = crate::env_config::tuning();
    let mailbox = Arc::new(OutboundMailbox::new(OutboundLimits {
        max_items: tuning.voice_ws_outbound_items.max(OUTBOUND_ITEMS),
        max_audio_bytes: tuning.voice_ws_outbound_audio_bytes,
        max_audio_ms: tuning.voice_ws_outbound_audio_ms,
    }));
    let writer = spawn_writer(ws_tx, mailbox.clone(), 1011);
    let out = Outbound {
        mailbox: mailbox.clone(),
        ids: ids.clone(),
    };
    let (finished_tx, mut finished_rx) = mpsc::unbounded_channel();
    let config = SessionConfig::new(
        ids.next("sess"),
        requested_model,
        profile.default_voice.clone(),
        DEFAULT_VOICE_AGENT_SYSTEM_PROMPT.to_string(),
        unix_now() + SESSION_LIFETIME.as_secs(),
    );
    let profile_for_history = profile.clone();
    let config_instructions = config.instructions.clone();
    let mut session = Session {
        state: state.clone(),
        correlation_id,
        kind,
        out,
        input: input_for(&config),
        profile,
        conversation_id: ids.next("conv"),
        config,
        items: Arc::default(),
        frame_seq: 0,
        input_ms: 0.0,
        speech: None,
        live: None,
        responses: Vec::new(),
        next_generation: 0,
        pending_auto_response: false,
        voice_locked: false,
        custom_voice: None,
        model_leases: SessionModelLeases::default(),
        pending_transcripts: Arc::default(),
        lane: Arc::default(),
        mcp: SharedMcp::default(),
        mcp_announced: Vec::new(),
        finished_tx,
        history: (super::persist::enabled() && kind == SessionKind::Realtime).then(|| {
            History::new(
                state.clone(),
                profile_for_history.clone(),
                config_instructions.clone(),
            )
        }),
        last_activity: Instant::now(),
        idle_fired: false,
    };
    if kind == SessionKind::Transcription {
        session.config.transcription = Some(Default::default());
    }
    // An ephemeral key's session config applies before the session starts.
    let bound_error = match bound_session.as_ref() {
        Some(bound) => session.apply_session(bound).await.err(),
        None => None,
    };
    session.announce();
    if let Some(err) = bound_error {
        session.send_error(err);
    }
    session.warm_models();
    state.runtime.record_voice_session_started();
    let started = Instant::now();

    let mut tick = tokio::time::interval(SESSION_ACTOR_TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    tick.tick().await;
    loop {
        tokio::select! {
            message = ws_rx.next() => {
                let Some(Ok(message)) = message else { break };
                match message {
                    Message::Text(text) => session.on_text(text.as_str()).await,
                    Message::Binary(_) => session.send_error(RealtimeError::invalid(
                        "invalid_event",
                        "Binary messages are not part of the Realtime protocol; send JSON events.",
                    )),
                    Message::Ping(payload) => {
                        let _ = mailbox.enqueue(crate::app::realtime_transport::OutboundItem {
                            message: Message::Pong(payload).into(),
                            class: crate::app::realtime_transport::OutboundClass::Droppable,
                            turn_generation: None,
                        });
                    }
                    Message::Pong(_) => {}
                    Message::Close(_) => {
                        // Answer the client's close, as WebSocket clients
                        // (Pipecat's `websockets`) expect.
                        let _ = mailbox.enqueue(crate::app::realtime_transport::OutboundItem {
                            message: Message::Close(Some(CloseFrame {
                                code: 1000,
                                reason: "".into(),
                            }))
                            .into(),
                            class: crate::app::realtime_transport::OutboundClass::Critical,
                            turn_generation: None,
                        });
                        break;
                    }
                }
            }
            Some((response_id, summary)) = finished_rx.recv() => session.on_response_finished(&response_id, summary),
            _ = tick.tick() => session.on_tick(),
        }
    }

    for response in &session.responses {
        response.control.cancel("session_closed");
    }
    for response in session.responses.drain(..) {
        if tokio::time::timeout(Duration::from_millis(750), response.task)
            .await
            .is_err()
        {
            debug!("realtime response did not stop in time at session close");
        }
    }
    if let Some(history) = session.history.as_ref() {
        history.close().await;
    }
    state.runtime.record_voice_session_closed();
    state
        .runtime
        .record_voice_latency(izwi_core::VoiceLatency::SessionDuration, started.elapsed());
    mailbox.close();
    let _ = writer.await;
}

impl Session {
    fn send_error(&self, error: RealtimeError) {
        debug!(code = ?error.code, message = %error.message, "realtime client error");
        self.out.value(error.to_event(&self.out.ids));
    }

    /// `session.created` and `conversation.created`.
    fn announce(&self) {
        self.out
            .event("session.created", json!({"session": self.session_json()}));
        if self.kind == SessionKind::Realtime {
            self.out.event(
                "conversation.created",
                json!({"conversation": {"id": self.conversation_id, "object": "realtime.conversation"}}),
            );
        }
    }

    fn session_json(&self) -> Value {
        match self.kind {
            SessionKind::Realtime => self.config.to_json(),
            SessionKind::Transcription => json!({
                "type": "transcription",
                "object": "realtime.transcription_session",
                "id": self.config.id,
                "expires_at": self.config.expires_at,
                "include": self.config.include,
                "audio": {"input": self.config.to_json()["audio"]["input"]},
            }),
        }
    }

    /// Load the session's models in the background, so the first turn does
    /// not wait for a cold load.
    fn warm_models(&self) {
        let mut models = match self.kind {
            SessionKind::Realtime => self
                .profile
                .models()
                .into_iter()
                .map(|(_, _, variant)| variant)
                .collect(),
            SessionKind::Transcription => Vec::new(),
        };
        if let Ok(model) = self.transcription_model() {
            models.push(izwi_core::resolve_asr_model_variant(Some(&model)));
        }
        let runtime = self.state.runtime.clone();
        let leases = self.model_leases.clone();
        tokio::spawn(async move {
            for variant in models {
                if let Err(err) = leases.hold(&runtime, variant).await {
                    warn!(model = %variant.dir_name(), "realtime model warm-up failed: {err}");
                }
            }
        });
    }

    fn transcription_model(&self) -> Result<String, String> {
        transcription_model(
            &self.profile,
            self.config
                .transcription
                .as_ref()
                .and_then(|transcription| transcription.model.as_deref()),
        )
    }

    async fn on_text(&mut self, text: &str) {
        let parsed = match parse_client_event(text) {
            Ok(parsed) => parsed,
            Err(err) => return self.send_error(err),
        };
        let event_id = parsed.event_id.clone();
        if let Err(err) = self.on_event(parsed.event).await {
            self.send_error(err.with_event_id(event_id));
        }
    }

    async fn on_event(&mut self, event: ClientEvent) -> Result<(), RealtimeError> {
        match event {
            ClientEvent::SessionUpdate { session } => self.on_session_update(&session).await,
            ClientEvent::InputAudioAppend { audio } => self.on_append(&audio),
            ClientEvent::InputAudioCommit => self.on_commit(),
            ClientEvent::InputAudioClear => {
                match &mut self.input {
                    Input::Manual(buffer) => buffer.clear(),
                    Input::Vad(input) => input.clear(),
                }
                self.speech = None;
                self.live = None;
                self.out.event("input_audio_buffer.cleared", json!({}));
                Ok(())
            }
            ClientEvent::ItemCreate {
                previous_item_id,
                item,
            } => self.on_item_create(previous_item_id.as_deref(), &item),
            ClientEvent::ItemRetrieve { item_id } => self.on_item_retrieve(&item_id),
            ClientEvent::ItemTruncate {
                item_id,
                content_index,
                audio_end_ms,
            } => self.on_item_truncate(&item_id, content_index, audio_end_ms),
            ClientEvent::ItemDelete { item_id } => {
                if lock_items(&self.items).delete(&item_id).is_none() {
                    return Err(RealtimeError::invalid(
                        "item_delete_invalid_item_id",
                        format!("Item with item_id '{item_id}' not found."),
                    )
                    .with_param("item_id"));
                }
                self.out
                    .event("conversation.item.deleted", json!({"item_id": item_id}));
                Ok(())
            }
            ClientEvent::ResponseCreate { response } => self.on_response_create(&response),
            ClientEvent::ResponseCancel { response_id } => {
                self.on_response_cancel(response_id.as_deref())
            }
            ClientEvent::OutputAudioBufferClear => Err(RealtimeError::invalid(
                "unsupported_on_websocket",
                "output_audio_buffer.clear is for WebRTC and SIP connections; over a WebSocket, \
                 send response.cancel and conversation.item.truncate.",
            )),
        }
    }

    async fn on_session_update(&mut self, update: &Value) -> Result<(), RealtimeError> {
        self.apply_session(update).await?;
        self.out
            .event("session.updated", json!({"session": self.session_json()}));
        self.announce_mcp_tools();
        Ok(())
    }

    /// List the tools of the session's MCP servers when they change:
    /// `mcp_list_tools.in_progress`, then per server an `mcp_list_tools`
    /// item and `mcp_list_tools.completed`, or `mcp_list_tools.failed`.
    /// Runs beside the session; a response waits for it.
    fn announce_mcp_tools(&mut self) {
        let configs = mcp_configs(&self.config.tools);
        if configs == self.mcp_announced {
            return;
        }
        self.mcp_announced = configs.clone();
        if configs.is_empty() {
            return;
        }
        let (out, shared, tools) = (
            self.out.clone(),
            self.mcp.clone(),
            self.config.tools.clone(),
        );
        let listings: Vec<(String, String)> = configs
            .iter()
            .map(|config| (config.label.clone(), out.ids.next("item")))
            .collect();
        for (_, item_id) in &listings {
            out.event("mcp_list_tools.in_progress", json!({"item_id": item_id}));
        }
        tokio::spawn(async move {
            match mcp_tools_for(&shared, &tools).await {
                Ok(mcp) => {
                    for (label, item_id) in listings {
                        let tools: Vec<Value> = mcp
                            .tools_of(&label)
                            .into_iter()
                            .map(|tool| {
                                json!({
                                    "name": tool.name,
                                    "description": tool.description,
                                    "input_schema": tool.input_schema,
                                })
                            })
                            .collect();
                        out.event(
                            "conversation.item.done",
                            json!({"previous_item_id": null, "item": {
                                "id": item_id, "type": "mcp_list_tools",
                                "server_label": label, "tools": tools,
                            }}),
                        );
                        out.event("mcp_list_tools.completed", json!({"item_id": item_id}));
                    }
                }
                Err(err) => {
                    warn!("realtime MCP tool listing failed: {err}");
                    for (_, item_id) in listings {
                        out.event("mcp_list_tools.failed", json!({"item_id": item_id}));
                    }
                }
            }
        });
    }

    /// Merge a session config into the session's.
    async fn apply_session(&mut self, update: &Value) -> Result<(), RealtimeError> {
        let mut update = update.clone();
        if self.kind == SessionKind::Transcription {
            // A transcription session takes the realtime shape's input half.
            if update.get("type").and_then(Value::as_str) == Some("transcription") {
                update["type"] = json!("realtime");
            }
            if let Some(object) = update.as_object_mut() {
                object.retain(|key, _| key == "type" || key == "audio" || key == "include");
            }
        }
        let outcome = self.config.apply_update(&update, self.voice_locked)?;
        for ignored in &outcome.ignored {
            debug!("realtime session.update ignored {ignored}");
        }
        if self.kind == SessionKind::Transcription && self.config.transcription.is_none() {
            self.config.transcription = Some(Default::default());
        }
        if outcome.input_changed {
            self.input = input_for(&self.config);
            self.speech = None;
            self.live = None;
        }
        self.custom_voice = match &self.config.voice {
            Voice::Custom(id) => Some(
                resolve_custom_voice(&self.state, &self.profile, id)
                    .await
                    .unwrap_or_else(|reason| {
                        warn!(voice = %id, "realtime custom voice not used: {reason}");
                        TtsVoice::speaker(None)
                    }),
            ),
            // An uploaded voice by name, for a TTS model that clones and has
            // no built-in voices.
            Voice::Named(name) => resolve_named_voice(&self.state, &self.profile, name).await,
        };
        Ok(())
    }

    fn on_append(&mut self, audio: &str) -> Result<(), RealtimeError> {
        let samples = decode_input(audio, self.config.input_format)?;
        if samples.is_empty() {
            return Ok(());
        }
        let rate = self.config.input_format.sample_rate();
        let chunk_start_ms = self.input_ms;
        self.input_ms += samples.len() as f64 * 1_000.0 / f64::from(rate);
        match &mut self.input {
            Input::Manual(buffer) => {
                if buffer.len() + samples.len() > MAX_UTTERANCE_PCM16_SAMPLES {
                    return Err(RealtimeError::invalid(
                        "input_audio_buffer_full",
                        "The input audio buffer is full; commit or clear it.",
                    ));
                }
                let first = buffer.is_empty();
                buffer.extend_from_slice(&samples);
                // A manual turn runs from its first audio to the commit.
                if !first {
                    if let Some(live) = self.live.as_ref() {
                        live.push(samples);
                    }
                } else if let Some(model_id) = self.live_transcription_model() {
                    let item_id = self.out.ids.next("item");
                    self.start_live(model_id, item_id, samples);
                }
                Ok(())
            }
            Input::Vad(input) => {
                self.frame_seq = self.frame_seq.wrapping_add(1).max(1);
                let result = input
                    .handle_pcm16_frame(self.frame_seq, rate, &pcm16_bytes(&samples))
                    .map_err(RealtimeError::server)?;
                // A live transcript starts once speech is confirmed, from
                // the utterance so far, so its deltas follow `speech_started`
                // and noise that never becomes a turn is not transcribed.
                let heard = match (result.barge_in, input.active.as_ref()) {
                    (true, Some(active)) => Some(active.samples_i16.clone()),
                    _ => None,
                };
                if let Some(start) = result.speech_start {
                    let pre_roll_ms = start.pre_roll_samples as f64 * 1_000.0 / f64::from(rate);
                    self.speech = Some(Speech {
                        item_id: self.out.ids.next("item"),
                        start_ms: (chunk_start_ms - pre_roll_ms).max(0.0) as u64,
                        announced: false,
                    });
                }
                if result.barge_in {
                    self.announce_speech();
                }
                if let Some(heard) = heard {
                    let speech = self.speech.as_ref().map(|speech| speech.item_id.clone());
                    if let (Some(item_id), Some(model_id)) =
                        (speech, self.live_transcription_model())
                    {
                        self.start_live(model_id, item_id, heard);
                    }
                } else if let Some(live) = self.live.as_ref() {
                    live.push(samples);
                }
                if let Some(outcome) = result.utterance_end {
                    self.on_utterance_end(outcome, true);
                }
                Ok(())
            }
        }
    }

    /// `input_audio_buffer.speech_started`, once speech is confirmed; with
    /// `interrupt_response` it cancels the response in progress.
    fn announce_speech(&mut self) {
        if self.speech.is_none() {
            self.speech = Some(Speech {
                item_id: self.out.ids.next("item"),
                start_ms: self.input_ms as u64,
                announced: false,
            });
        }
        let Some(speech) = self.speech.as_mut().filter(|speech| !speech.announced) else {
            return;
        };
        speech.announced = true;
        self.out.event(
            "input_audio_buffer.speech_started",
            json!({"audio_start_ms": speech.start_ms, "item_id": speech.item_id}),
        );
        self.last_activity = Instant::now();
        self.idle_fired = false;
        let interrupt = self
            .config
            .turn_detection
            .as_ref()
            .is_none_or(|turn_detection| turn_detection.interrupt_response());
        if interrupt {
            if let Some(index) = self.active_default_response() {
                self.state.runtime.record_voice_barge_in();
                self.cancel_response(index, "turn_detected");
            }
        }
    }

    fn on_utterance_end(&mut self, outcome: UtteranceOutcome, auto_response: bool) {
        match outcome {
            UtteranceOutcome::Finalized(ended, _) => {
                self.announce_speech();
                let speech = self.speech.take();
                let item_id = speech
                    .map(|speech| speech.item_id)
                    .unwrap_or_else(|| self.out.ids.next("item"));
                self.out.event(
                    "input_audio_buffer.speech_stopped",
                    json!({"audio_end_ms": self.input_ms as u64, "item_id": item_id}),
                );
                let create = self
                    .config
                    .turn_detection
                    .as_ref()
                    .is_some_and(|turn_detection| turn_detection.create_response());
                let respond = auto_response && create && self.kind == SessionKind::Realtime;
                // The reply that hears this turn transcribes it (LFM3-Audio).
                let defer = respond && self.replies_transcribe();
                self.commit_audio(item_id, ended.samples, ended.sample_rate, defer);
                if respond {
                    if self.active_default_response().is_some()
                        || self
                            .responses
                            .iter()
                            .any(|response| response.default_conversation)
                    {
                        self.pending_auto_response = true;
                    } else {
                        self.start_auto_response();
                    }
                }
            }
            UtteranceOutcome::Rejected(_) => {
                self.live = None;
                self.state.runtime.record_voice_speech_rejected();
                if let Some(speech) = self.speech.take().filter(|speech| speech.announced) {
                    self.out.event(
                        "input_audio_buffer.speech_stopped",
                        json!({"audio_end_ms": self.input_ms as u64, "item_id": speech.item_id}),
                    );
                }
            }
        }
        self.last_activity = Instant::now();
    }

    /// Committed user speech becomes a user item, and is transcribed.
    fn commit_audio(
        &mut self,
        item_id: String,
        samples: Vec<i16>,
        sample_rate: u32,
        defer_transcript: bool,
    ) {
        let audio_secs = samples.len() as f32 / sample_rate.max(1) as f32;
        let wav_bytes = match wav(&samples, sample_rate) {
            Ok(bytes) => bytes,
            Err(err) => {
                return self.send_error(RealtimeError::server(format!(
                    "Could not encode the committed audio: {err}"
                )))
            }
        };
        let (ready_tx, ready_rx) = watch::channel(false);
        let mut item = Item::new(
            item_id.clone(),
            ItemKind::Message {
                role: Role::User,
                content: vec![ContentPart::InputAudio { transcript: None }],
            },
            ItemStatus::Completed,
        );
        item.audio = Some(ItemAudio {
            samples: Arc::new(samples),
            sample_rate,
        });
        item.transcript_ready = Some(ready_rx);
        let json = item.to_json();
        let previous = match lock_items(&self.items).insert(item, &Position::End) {
            Ok(previous) => previous,
            Err(err) => return self.send_error(err),
        };
        self.out.event(
            "input_audio_buffer.committed",
            json!({"previous_item_id": previous, "item_id": item_id}),
        );
        if self.kind == SessionKind::Realtime {
            self.out.event(
                "conversation.item.added",
                json!({"previous_item_id": previous, "item": json}),
            );
            self.out.event(
                "conversation.item.done",
                json!({"previous_item_id": previous, "item": json}),
            );
        }
        if defer_transcript {
            self.pending_transcripts
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(
                    item_id,
                    PendingTranscript {
                        ready: ready_tx,
                        emit: self.config.transcription.is_some(),
                        audio_secs,
                    },
                );
            return;
        }
        // A turn transcribed as it was spoken ends its live transcript;
        // any other is transcribed now.
        let live = self.live.take().filter(|live| live.item_id == item_id);
        let transcription = self.transcription(item_id, wav_bytes, ready_tx);
        match live {
            Some(live) => live.finish(transcription),
            None => transcription.spawn(),
        }
    }

    /// The model that transcribes this session's turns as they are spoken,
    /// when it is one that can.
    fn live_transcription_model(&self) -> Option<String> {
        let model_id = self.transcription_model().ok()?;
        transcribes_live(&self.profile, self.kind == SessionKind::Realtime, &model_id)
            .then_some(model_id)
    }

    /// Transcribe the turn that becomes `item_id` as it is spoken, from the
    /// audio `heard` so far.
    fn start_live(&mut self, model_id: String, item_id: String, heard: Vec<i16>) {
        let transcription = self.config.transcription.as_ref();
        self.live = Some(
            LiveTranscription {
                state: self.state.clone(),
                item_id,
                model_id,
                language: transcription.and_then(|config| config.language.clone()),
                prompt: transcription.and_then(|config| config.prompt.clone()),
                emit: transcription.is_some(),
                out: self.out.clone(),
                model_leases: self.model_leases.clone(),
                sample_rate: self.config.input_format.sample_rate(),
            }
            .spawn(heard),
        );
    }

    /// Whether replies transcribe the turn they hear (LFM3-Audio), so a
    /// turn's transcription waits for its reply.
    fn replies_transcribe(&self) -> bool {
        self.kind == SessionKind::Realtime
            && matches!(&self.profile.mode, super::profiles::ProfileMode::Unified { s2s, .. }
                if s2s.family() == izwi_core::ModelFamily::Lfm3Audio)
    }

    /// Transcribe the turns a reply left untranscribed (it failed or was
    /// cancelled), now through the ASR route.
    fn transcribe_left_over(&mut self) {
        let left: Vec<(String, PendingTranscript)> = self
            .pending_transcripts
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .drain()
            .collect();
        for (item_id, pending) in left {
            let audio = lock_items(&self.items)
                .get(&item_id)
                .and_then(|item| item.audio.clone());
            match audio.map(|audio| wav(&audio.samples, audio.sample_rate)) {
                Some(Ok(wav_bytes)) => self.transcribe(item_id, wav_bytes, pending.ready),
                _ => {
                    let _ = pending.ready.send(true);
                }
            }
        }
    }

    fn transcribe(&self, item_id: String, wav_bytes: Vec<u8>, ready: watch::Sender<bool>) {
        self.transcription(item_id, wav_bytes, ready).spawn();
    }

    /// The turn-based transcription of a committed item.
    fn transcription(
        &self,
        item_id: String,
        wav_bytes: Vec<u8>,
        ready: watch::Sender<bool>,
    ) -> Transcription {
        let model_id = self.transcription_model().unwrap_or_else(|err| {
            warn!("realtime transcription model not used: {err}");
            transcription_model(&self.profile, None).unwrap_or_default()
        });
        let transcription = self.config.transcription.clone();
        Transcription {
            state: self.state.clone(),
            correlation_id: self.correlation_id.clone(),
            out: self.out.clone(),
            items: self.items.clone(),
            model_leases: self.model_leases.clone(),
            item_id,
            wav: wav_bytes,
            model_id,
            language: transcription
                .as_ref()
                .and_then(|config| config.language.clone()),
            prompt: transcription
                .as_ref()
                .and_then(|config| config.prompt.clone()),
            emit: transcription.is_some(),
            emit_deltas: true,
            ready,
            lane: self.profile.is_unified().then(|| self.lane.clone()),
        }
    }

    fn on_commit(&mut self) -> Result<(), RealtimeError> {
        let rate = self.config.input_format.sample_rate();
        let empty = |ms: f64| {
            RealtimeError::invalid(
                "input_audio_buffer_commit_empty",
                format!(
                    "Error committing input audio buffer: buffer too small. Expected at least \
                     100ms of audio, but buffer only has {ms:.2}ms of audio."
                ),
            )
        };
        match &mut self.input {
            Input::Manual(buffer) => {
                let ms = buffer.len() as f64 * 1_000.0 / f64::from(rate);
                if ms < MIN_COMMIT_MS {
                    return Err(empty(ms));
                }
                let samples = std::mem::take(buffer);
                // Live deltas already name the item this turn becomes.
                let item_id = match self.live.as_ref() {
                    Some(live) => live.item_id.clone(),
                    None => self.out.ids.next("item"),
                };
                self.commit_audio(item_id, samples, rate, false);
                self.last_activity = Instant::now();
                Ok(())
            }
            Input::Vad(input) => match input.commit_active().map_err(RealtimeError::server)? {
                Some(outcome @ UtteranceOutcome::Finalized(..)) => {
                    self.on_utterance_end(outcome, false);
                    Ok(())
                }
                Some(outcome @ UtteranceOutcome::Rejected(_)) => {
                    self.on_utterance_end(outcome, false);
                    Err(empty(0.0))
                }
                None => Err(empty(0.0)),
            },
        }
    }

    fn on_item_create(
        &mut self,
        previous_item_id: Option<&str>,
        value: &Value,
    ) -> Result<(), RealtimeError> {
        let mut item = Item::from_client(value, || self.out.ids.next("item"))?;
        // Spoken input in an item (a client replaying history) is kept to be
        // heard, and transcribed when it carries no transcript.
        let audio = value
            .get("content")
            .and_then(Value::as_array)
            .and_then(|parts| {
                parts
                    .iter()
                    .find_map(|part| part.get("audio").and_then(Value::as_str))
            })
            .filter(|audio| !audio.is_empty());
        let mut transcribe = None;
        if let (Some(audio), Some(Role::User)) = (audio, item.role()) {
            let samples = decode_input(audio, self.config.input_format)?;
            let rate = self.config.input_format.sample_rate();
            if item.awaits_transcript() && !samples.is_empty() {
                let (ready_tx, ready_rx) = watch::channel(false);
                item.transcript_ready = Some(ready_rx);
                transcribe = Some((
                    wav(&samples, rate).map_err(RealtimeError::server)?,
                    ready_tx,
                ));
            }
            item.audio = Some(ItemAudio {
                samples: Arc::new(samples),
                sample_rate: rate,
            });
        }
        let item_id = item.id.clone();
        let json = item.to_json();
        let previous = lock_items(&self.items).insert(item, &Position::parse(previous_item_id))?;
        self.out.event(
            "conversation.item.added",
            json!({"previous_item_id": previous, "item": json}),
        );
        self.out.event(
            "conversation.item.done",
            json!({"previous_item_id": previous, "item": json}),
        );
        if let Some((wav_bytes, ready)) = transcribe {
            self.transcribe(item_id, wav_bytes, ready);
        }
        Ok(())
    }

    fn on_item_retrieve(&self, item_id: &str) -> Result<(), RealtimeError> {
        let item = lock_items(&self.items)
            .get(item_id)
            .cloned()
            .ok_or_else(|| {
                RealtimeError::invalid(
                    "item_retrieve_invalid_item_id",
                    format!("Item with item_id '{item_id}' not found."),
                )
                .with_param("item_id")
            })?;
        let mut json = item.to_json();
        if let Some(audio) = item.audio.as_ref() {
            if let Some(part) = json["content"].get_mut(0) {
                part["audio"] = json!(base64(&pcm16_bytes(&audio.samples)));
            }
        }
        self.out
            .event("conversation.item.retrieved", json!({"item": json}));
        Ok(())
    }

    /// The client played only `audio_end_ms` of an assistant item: the item
    /// keeps the transcript of what was heard. A position past the end is
    /// clamped (Pipecat treats an error as fatal), and an unknown item is
    /// ignored.
    fn on_item_truncate(
        &mut self,
        item_id: &str,
        content_index: u64,
        audio_end_ms: i64,
    ) -> Result<(), RealtimeError> {
        let audio_end_ms = {
            let mut items = lock_items(&self.items);
            let Some(item) = items.get_mut(item_id) else {
                debug!(%item_id, "realtime truncate of an unknown item ignored");
                return Ok(());
            };
            if item.role() != Some(Role::Assistant) {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    "Only assistant message items can be truncated.",
                )
                .with_param("item_id"));
            }
            let requested = audio_end_ms.max(0) as u64;
            // A finished reply knows its length; one still streaming does not yet.
            let audio_end_ms = if item.audio_ms > 0 {
                requested.min(item.audio_ms)
            } else {
                requested
            };
            let full = item.text().unwrap_or_default();
            let heard = match item.reply.as_ref() {
                Some(control) => {
                    let heard = control.truncate(audio_end_ms);
                    if control.reply().ledger.audio_ms() > 0.0 {
                        heard
                    } else {
                        heard_share(&full, audio_end_ms, item.audio_ms)
                    }
                }
                None => heard_share(&full, audio_end_ms, item.audio_ms),
            };
            if let ItemKind::Message { content, .. } = &mut item.kind {
                *content = vec![ContentPart::OutputAudio { transcript: heard }];
            }
            audio_end_ms
        };
        self.out.event(
            "conversation.item.truncated",
            json!({"item_id": item_id, "content_index": content_index, "audio_end_ms": audio_end_ms}),
        );
        Ok(())
    }

    fn active_default_response(&self) -> Option<usize> {
        self.responses
            .iter()
            .position(|response| response.default_conversation && !response.control.is_cancelled())
    }

    fn cancel_response(&mut self, index: usize, reason: &str) {
        let response = &self.responses[index];
        response.control.cancel(reason);
        // The response's queued audio goes at once; its closing events and
        // `response.done` still follow.
        if let Some(generation) = response.generation {
            self.out.mailbox.cutoff_turn(generation);
        }
        self.state.runtime.record_voice_interruption();
    }

    fn on_response_cancel(&mut self, response_id: Option<&str>) -> Result<(), RealtimeError> {
        let index = match response_id {
            Some(id) => self
                .responses
                .iter()
                .position(|response| response.id == id && !response.control.is_cancelled()),
            None => self.active_default_response(),
        };
        match index {
            Some(index) => {
                self.cancel_response(index, "client_cancelled");
                Ok(())
            }
            // LiveKit ignores errors whose message starts "Cancellation
            // failed"; Pipecat tolerates this code.
            None => Err(RealtimeError::invalid(
                "response_cancel_not_active",
                "Cancellation failed: no active response found",
            )),
        }
    }

    fn on_response_create(&mut self, params: &Value) -> Result<(), RealtimeError> {
        if self.kind == SessionKind::Transcription {
            return Err(RealtimeError::invalid(
                "invalid_session_type",
                "A transcription session creates no responses.",
            ));
        }
        let plan = self.plan_response(params)?;
        if plan.default_conversation {
            if let Some(active) = self
                .responses
                .iter()
                .find(|response| response.default_conversation)
            {
                return Err(RealtimeError::invalid(
                    "conversation_already_has_active_response",
                    format!(
                        "Conversation already has an active response in progress: {}. Wait until \
                         the response is finished before creating a new one.",
                        active.id
                    ),
                ));
            }
        } else if self
            .responses
            .iter()
            .any(|response| !response.default_conversation)
        {
            return Err(RealtimeError::invalid(
                "conversation_already_has_active_response",
                "An out-of-band response is already in progress; wait for it to finish.",
            ));
        }
        self.start_response(plan);
        Ok(())
    }

    fn start_auto_response(&mut self) {
        self.pending_auto_response = false;
        match self.plan_response(&json!({})) {
            Ok(plan) => self.start_response(plan),
            Err(err) => self.send_error(err),
        }
    }

    /// A response's plan: `response.create`'s overrides over the session.
    fn plan_response(&self, params: &Value) -> Result<ResponsePlan, RealtimeError> {
        let default_conversation = match params.get("conversation").and_then(Value::as_str) {
            None | Some("auto") => true,
            Some("none") => false,
            Some(other) => {
                return Err(RealtimeError::invalid(
                    "invalid_value",
                    format!("Invalid conversation '{other}': use \"auto\" or \"none\"."),
                )
                .with_param("response.conversation"))
            }
        };
        let input = match params.get("input") {
            None | Some(Value::Null) => None,
            Some(Value::Array(entries)) => {
                let items = lock_items(&self.items);
                Some(
                    entries
                        .iter()
                        .map(|entry| {
                            if entry.get("type").and_then(Value::as_str) == Some("item_reference") {
                                let id =
                                    entry.get("id").and_then(Value::as_str).unwrap_or_default();
                                items.get(id).cloned().ok_or_else(|| {
                                    RealtimeError::invalid(
                                        "item_reference_not_found",
                                        format!("Item with id '{id}' not found."),
                                    )
                                    .with_param("response.input")
                                })
                            } else {
                                Item::from_client(entry, || self.out.ids.next("item"))
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            Some(_) => {
                return Err(
                    RealtimeError::invalid("invalid_type", "'input' must be an array.")
                        .with_param("response.input"),
                )
            }
        };
        let instructions = match params.get("instructions") {
            Some(Value::String(instructions)) => instructions.clone(),
            _ => self.config.instructions.clone(),
        };
        let modality = match params
            .get("output_modalities")
            .or_else(|| params.get("modalities"))
        {
            Some(value) => Modality::parse(value, "response.output_modalities")?,
            None => self.config.modality,
        };
        let tools = match params.get("tools") {
            Some(value) => parse_tools(value, "response.tools")?,
            None => self.config.tools.clone(),
        };
        let max_output_tokens = match params.get("max_output_tokens") {
            Some(value) => parse_max_output_tokens(value, "response.max_output_tokens")?,
            None => self.config.max_output_tokens,
        };
        let metadata = parse_metadata(params.get("metadata").unwrap_or(&Value::Null))?;
        let output = params.get("audio").and_then(|audio| audio.get("output"));
        let output_format = match output.and_then(|output| output.get("format")) {
            Some(format) => {
                let mut probe = self.config.clone();
                probe.apply_update(&json!({"audio": {"output": {"format": format}}}), false)?;
                probe.output_format
            }
            None => self.config.output_format,
        };
        let voice = match output
            .and_then(|output| output.get("voice"))
            .and_then(Value::as_str)
        {
            Some(voice) if !self.voice_locked => Voice::Named(voice.to_string()),
            _ => self.config.voice.clone(),
        };
        let tts_voice = match &voice {
            // The session's own name resolved to an uploaded voice.
            Voice::Named(_) if voice == self.config.voice && self.custom_voice.is_some() => {
                self.custom_voice.clone().unwrap_or_default()
            }
            Voice::Named(name) => TtsVoice::speaker(speaker_for(self.profile.speech_model(), name)),
            Voice::Custom(_) => self.custom_voice.clone().unwrap_or_default(),
        };
        Ok(ResponsePlan {
            id: self.out.ids.next("resp"),
            conversation_id: self.conversation_id.clone(),
            default_conversation,
            input,
            instructions,
            modality,
            tools,
            tool_choice: params
                .get("tool_choice")
                .cloned()
                .unwrap_or_else(|| self.config.tool_choice.clone()),
            max_output_tokens,
            voice: tts_voice,
            voice_echo: voice.to_json(),
            output_format: if modality == Modality::Audio {
                output_format
            } else {
                AudioFormat::Pcm24k
            },
            metadata,
            audio_generation: None,
        })
    }

    fn start_response(&mut self, mut plan: ResponsePlan) {
        if plan.default_conversation {
            self.next_generation += 1;
            plan.audio_generation = Some(self.next_generation);
        }
        self.out.event(
            "response.created",
            json!({"response": plan.response_json("in_progress", Value::Null, Vec::new(), Value::Null)}),
        );
        let control = TurnControl::new(Instant::now());
        let context = ResponseContext {
            state: self.state.clone(),
            correlation_id: self.correlation_id.clone(),
            out: self.out.clone(),
            items: self.items.clone(),
            profile: self.profile.clone(),
            model_leases: self.model_leases.clone(),
            pending_transcripts: self.pending_transcripts.clone(),
            lane: self.lane.clone(),
            mcp: self.mcp.clone(),
        };
        let id = plan.id.clone();
        let generation = plan.audio_generation;
        let default_conversation = plan.default_conversation;
        let finished = self.finished_tx.clone();
        let task_control = control.clone();
        let task_id = id.clone();
        let task = tokio::spawn(async move {
            let summary = run_response(context, Arc::new(plan), task_control).await;
            let _ = finished.send((task_id, summary));
        });
        self.responses.push(ActiveResponse {
            id,
            control,
            task,
            generation,
            default_conversation,
        });
    }

    fn on_response_finished(&mut self, response_id: &str, summary: ResponseSummary) {
        let Some(index) = self
            .responses
            .iter()
            .position(|response| response.id == response_id)
        else {
            return;
        };
        let response = self.responses.remove(index);
        self.voice_locked |= summary.spoke;
        if let (Some(history), true) = (self.history.as_ref(), response.default_conversation) {
            let user_text = lock_items(&self.items)
                .items()
                .iter()
                .rev()
                .find(|item| item.role() == Some(Role::User))
                .and_then(Item::text);
            let speaker = match &self.config.voice {
                Voice::Named(name) => Some(name.clone()),
                Voice::Custom(id) => Some(id.clone()),
            };
            history.record_turn(user_text, summary.text.clone(), summary.status, speaker);
        }
        if !self
            .responses
            .iter()
            .any(|response| response.default_conversation)
        {
            self.transcribe_left_over();
        }
        // The conversation is idle once the reply has played out.
        let playing_until = response
            .control
            .inner
            .playback
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .playing_until;
        self.last_activity =
            playing_until.map_or_else(Instant::now, |until| until.max(Instant::now()));
        self.idle_fired = false;
        if self.pending_auto_response
            && response.default_conversation
            && !self
                .responses
                .iter()
                .any(|response| response.default_conversation)
        {
            self.start_auto_response();
        }
    }

    fn on_tick(&mut self) {
        if let Input::Vad(input) = &mut self.input {
            match input.on_tick() {
                Ok(Some(outcome)) => self.on_utterance_end(outcome, true),
                Ok(None) => {}
                Err(err) => self.send_error(RealtimeError::server(err)),
            }
        }
        self.check_idle();
    }

    /// `idle_timeout_ms`: no speech for that long after the conversation
    /// last moved commits the (silent) buffer and asks for a response.
    fn check_idle(&mut self) {
        let Some(idle_ms) = self
            .config
            .turn_detection
            .as_ref()
            .and_then(|turn_detection| turn_detection.idle_timeout_ms())
        else {
            return;
        };
        let speaking = match &self.input {
            Input::Vad(input) => input.has_active_utterance(),
            Input::Manual(_) => false,
        };
        if self.idle_fired
            || speaking
            || !self.responses.is_empty()
            || self.kind != SessionKind::Realtime
            || Instant::now() < self.last_activity + Duration::from_millis(u64::from(idle_ms))
        {
            return;
        }
        self.idle_fired = true;
        let end_ms = self.input_ms as u64;
        let start_ms = end_ms.saturating_sub(u64::from(idle_ms));
        let item_id = self.out.ids.next("item");
        self.out.event(
            "input_audio_buffer.timeout_triggered",
            json!({"audio_start_ms": start_ms, "audio_end_ms": end_ms, "item_id": item_id}),
        );
        let rate = self.config.input_format.sample_rate();
        let silence = vec![0i16; (rate / 10) as usize];
        let defer = self.replies_transcribe();
        self.commit_audio(item_id, silence, rate, defer);
        self.start_auto_response();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_truncated_transcript_keeps_the_heard_share() {
        assert_eq!(heard_share("one two three four", 500, 1_000), "one two");
        assert_eq!(heard_share("one two", 2_000, 1_000), "one two");
        assert_eq!(heard_share("one two", 0, 1_000), "");
        assert_eq!(heard_share("one two", 10, 0), "one two");
    }
}
