//! Bounded realtime WebSocket session client for the `izwi-realtime-v1`
//! worker subprotocol.
//!
//! One [`RealtimeSession`] binds exactly one realtime stage to one worker,
//! mirroring the HTTP invocation client's guarantees for the socket world:
//! the same endpoint rules (certificate-verified HTTPS or numeric loopback
//! HTTP), hard deadlines on every await, the announced session bounds
//! enforced before any frame leaves the process, and one-terminal-outcome
//! validation over the `InvocationEvent` vocabulary.
//!
//! Both protocol stages are served from one session type: the ASR-stream
//! stage pushes audio with [`RealtimeSession::send_audio`] and reads
//! transcript events with [`RealtimeSession::next_event`]; the TTS-stream
//! stage pushes utterance text with [`RealtimeSession::send_text`] and reads
//! synthesized audio with [`RealtimeSession::next_audio`]. Events and audio
//! interleave on the wire, so each read method buffers the other kind in a
//! bounded pending queue; a caller that never drains one kind fails closed
//! once that queue is full.
//!
//! Dropping a session without a terminal outcome leaves the worker session
//! running until its own deadline; the owner is responsible for an explicit
//! HTTP `cancel_attempt` in that case, exactly as for an uncertain HTTP
//! invocation.
//!
//! [`dial`] is the one worker WebSocket dialer: [`connect`] builds a
//! validating session on it, and the gateway relay uses it directly to
//! forward frames verbatim. HTTPS workers are verified against the public
//! web roots plus the configured private CA roots, presenting the mTLS
//! client identity when one is configured, exactly like the HTTP client.

use futures::{SinkExt, StreamExt};
use izwi_serving_protocol::{
    decode_realtime_audio_frame, encode_realtime_audio_frame, InvocationEvent, InvocationEventKind,
    RealtimeAudioSpec, RealtimeClientFrame, RealtimeServerFrame, RealtimeSessionAdmit,
    RealtimeSessionBounds, RealtimeStageInput, ServiceCredentials, TraceContext,
    MAX_REALTIME_INPUT_TEXT_BYTES, PROTOCOL_V1, REALTIME_SUBPROTOCOL, REALTIME_WS_PATH,
    SERVICE_AUTHORIZATION_HEADER, SERVICE_AUTH_SCHEME, SERVICE_CREDENTIAL_ID_HEADER,
    TRACEPARENT_HEADER,
};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;
use tokio_rustls::rustls;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest, http::HeaderValue, Error as WsError, Message,
};
use tokio_tungstenite::{Connector, MaybeTlsStream, WebSocketStream};

/// A worker session socket.
pub type WorkerSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Bound on events buffered while a caller drains audio instead. Events are
/// tiny and rare (accepted, terminal); overflowing means the caller is
/// misusing the session and must fail closed.
const MAX_PENDING_EVENTS: usize = 16;

/// Bounded deadlines for one realtime session client.
#[derive(Debug, Clone)]
pub struct RealtimeClientConfig {
    /// Handshake + upgrade deadline.
    pub connect_timeout: Duration,
    /// Deadline for the worker's `Admitted` frame after the upgrade.
    pub admit_timeout: Duration,
    /// Maximum idle wait between session frames once admitted.
    pub event_timeout: Duration,
    /// Deadline for any single outbound frame write.
    pub send_timeout: Duration,
    /// Private CA roots and mTLS identity for HTTPS workers.
    pub tls: crate::WorkerClientTlsConfig,
}

impl Default for RealtimeClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            admit_timeout: Duration::from_secs(10),
            event_timeout: Duration::from_secs(30),
            send_timeout: Duration::from_secs(5),
            tls: crate::WorkerClientTlsConfig::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RealtimeClientError {
    #[error("{0}")]
    InvalidConfiguration(&'static str),
    #[error("invalid realtime endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("realtime handshake failed")]
    Connect(#[source] WsError),
    #[error("worker did not select the izwi-realtime-v1 subprotocol")]
    Subprotocol,
    #[error("timed out waiting for the admitted frame")]
    AdmitTimeout,
    #[error("timed out waiting for a session frame")]
    EventTimeout,
    #[error("timed out writing a session frame")]
    SendTimeout,
    #[error("worker closed before admission with close code {0}")]
    ClosedBeforeAdmit(u16),
    #[error("worker closed the session with close code {0}")]
    Closed(u16),
    #[error("audio frame of {actual} bytes exceeds the negotiated bound {limit}")]
    AudioFrameTooLarge { actual: usize, limit: usize },
    #[error("session audio budget of {limit} bytes exhausted at {actual}")]
    AudioBudgetExhausted { actual: u64, limit: u64 },
    #[error("session text budget of {limit} bytes exhausted at {actual}")]
    TextBudgetExhausted { actual: u64, limit: u64 },
    #[error("{0}")]
    Protocol(String),
}

impl RealtimeClientError {
    /// True when the error proves the worker never admitted the session.
    pub fn proves_session_unaccepted(&self) -> bool {
        matches!(
            self,
            Self::InvalidConfiguration(_)
                | Self::InvalidEndpoint(_)
                | Self::Connect(_)
                | Self::Subprotocol
                | Self::ClosedBeforeAdmit(_)
                | Self::AdmitTimeout
        )
    }
}

/// The worker's `Admitted` echo: the negotiated session contract.
#[derive(Debug, Clone)]
pub struct RealtimeAdmission {
    pub session_id: String,
    pub attempt_id: String,
    pub worker_id: String,
    pub node_id: String,
    pub incarnation_id: String,
    pub deployment_id: String,
    pub bounds: RealtimeSessionBounds,
    /// The synthesized audio spec for stages that emit audio (TTS-stream);
    /// `None` for input-only stages.
    pub output_audio: Option<RealtimeAudioSpec>,
}

/// One synthesized audio frame delivered by the worker (TTS-stream stage).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealtimeAudioChunk {
    /// Worker-assigned monotonic sequence starting at 1.
    pub sequence: u32,
    /// True only on the terminal frame; payload may be empty there.
    pub is_final: bool,
    /// Raw PCM i16 LE payload at the admission's output audio sample rate.
    pub payload: Vec<u8>,
}

/// One admitted realtime session over a worker WebSocket.
pub struct RealtimeSession {
    stream: WorkerSocket,
    admission: RealtimeAdmission,
    expected_request_id: String,
    config: RealtimeClientConfig,
    /// The admitted stage emits audio to this client (TTS-stream).
    stage_emits_audio: bool,
    accepted_seen: bool,
    terminal_seen: bool,
    closed: bool,
    last_event_sequence: Option<u64>,
    next_audio_sequence: u32,
    audio_bytes: u64,
    text_bytes: u64,
    last_received_audio_sequence: Option<u32>,
    pending_events: VecDeque<InvocationEvent>,
    pending_audio: VecDeque<RealtimeAudioChunk>,
}

impl std::fmt::Debug for RealtimeSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RealtimeSession")
            .field("admission", &self.admission)
            .field("terminal_seen", &self.terminal_seen)
            .finish_non_exhaustive()
    }
}

/// A worker session that echoed the admit, before any session frame.
pub struct DialedSession {
    pub stream: WorkerSocket,
    pub admission: RealtimeAdmission,
    /// The worker's `Admitted` frame exactly as received, for relays that
    /// forward it verbatim.
    pub admitted_frame: String,
}

impl std::fmt::Debug for DialedSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DialedSession")
            .field("admission", &self.admission)
            .finish_non_exhaustive()
    }
}

/// Opens the session: upgrade, subprotocol negotiation, and the mandatory
/// first admit frame, resolved only once the worker's `Admitted` echo
/// arrives and matches the admit identity.
pub async fn connect(
    endpoint: &str,
    credentials: &ServiceCredentials,
    admit: RealtimeSessionAdmit,
    config: RealtimeClientConfig,
) -> Result<RealtimeSession, RealtimeClientError> {
    let DialedSession {
        stream, admission, ..
    } = dial(endpoint, credentials, &admit, None, &config).await?;
    Ok(RealtimeSession {
        stage_emits_audio: matches!(admit.input, RealtimeStageInput::TextStream { .. }),
        stream,
        admission,
        expected_request_id: admit.request_id.to_string(),
        config,
        accepted_seen: false,
        terminal_seen: false,
        closed: false,
        last_event_sequence: None,
        next_audio_sequence: 1,
        audio_bytes: 0,
        text_bytes: 0,
        last_received_audio_sequence: None,
        pending_events: VecDeque::new(),
        pending_audio: VecDeque::new(),
    })
}

/// Dials the worker's realtime socket, sends `admit`, and waits for the
/// matching `Admitted` echo. `trace` names the caller's span as the worker
/// session's parent.
pub async fn dial(
    endpoint: &str,
    credentials: &ServiceCredentials,
    admit: &RealtimeSessionAdmit,
    trace: Option<&TraceContext>,
    config: &RealtimeClientConfig,
) -> Result<DialedSession, RealtimeClientError> {
    let base =
        crate::validated_session_endpoint(endpoint).map_err(|rejection| match rejection {
            crate::EndpointRejection::Parse(error) => RealtimeClientError::InvalidEndpoint(error),
            crate::EndpointRejection::Policy(message) => {
                RealtimeClientError::InvalidConfiguration(message)
            }
        })?;
    if admit.validate().is_err() {
        return Err(RealtimeClientError::InvalidConfiguration(
            "realtime admission is not a valid izwi-realtime-v1 admit",
        ));
    }
    if base.scheme() != "https" && config.tls.is_configured() {
        return Err(RealtimeClientError::InvalidConfiguration(
            "custom TLS trust or mTLS identity requires an HTTPS worker endpoint",
        ));
    }
    let connector = tls_connector(&config.tls)?;
    let mut ws_url = base
        .join(REALTIME_WS_PATH.trim_start_matches('/'))
        .map_err(|error| RealtimeClientError::InvalidEndpoint(error.to_string()))?;
    let ws_scheme = match ws_url.scheme() {
        "https" => "wss",
        "http" => "ws",
        other => {
            return Err(RealtimeClientError::InvalidEndpoint(format!(
                "unsupported endpoint scheme {other}"
            )))
        }
    };
    ws_url.set_scheme(ws_scheme).map_err(|_| {
        RealtimeClientError::InvalidConfiguration(
            "endpoint scheme cannot be converted to a websocket scheme",
        )
    })?;
    let ws_url = ws_url.to_string();
    let mut request: tokio_tungstenite::tungstenite::http::Request<()> =
        ws_url.into_client_request().map_err(|error| {
            RealtimeClientError::InvalidEndpoint(format!("websocket request: {error}"))
        })?;
    let headers = request.headers_mut();
    headers.insert(
        "sec-websocket-protocol",
        HeaderValue::from_static(REALTIME_SUBPROTOCOL),
    );
    let authorization = HeaderValue::from_str(&format!(
        "{SERVICE_AUTH_SCHEME} {}",
        credentials.bearer_token.expose_secret()
    ))
    .map_err(|_| {
        RealtimeClientError::InvalidConfiguration("bearer token is not a valid header value")
    })?;
    headers.insert(SERVICE_AUTHORIZATION_HEADER, authorization);
    let credential = HeaderValue::from_str(credentials.credential_id.as_str()).map_err(|_| {
        RealtimeClientError::InvalidConfiguration("credential id is not a valid header value")
    })?;
    headers.insert(SERVICE_CREDENTIAL_ID_HEADER, credential);
    if let Some(trace) = trace {
        let traceparent = HeaderValue::from_str(&trace.header_value()).map_err(|_| {
            RealtimeClientError::InvalidConfiguration("traceparent is not a valid header value")
        })?;
        headers.insert(TRACEPARENT_HEADER, traceparent);
    }

    let handshake =
        tokio_tungstenite::connect_async_tls_with_config(request, None, false, connector);
    let (mut stream, response) = tokio::time::timeout(config.connect_timeout, handshake)
        .await
        .map_err(|_| RealtimeClientError::Connect(WsError::ConnectionClosed))?
        .map_err(RealtimeClientError::Connect)?;
    let selected = response
        .headers()
        .get("sec-websocket-protocol")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.trim().eq_ignore_ascii_case(REALTIME_SUBPROTOCOL));
    if selected != Some(true) {
        return Err(RealtimeClientError::Subprotocol);
    }

    send_control_frame(
        &mut stream,
        config.send_timeout,
        &RealtimeClientFrame::Admit {
            admit: Box::new(admit.clone()),
        },
    )
    .await?;

    // The admitted echo is the first server frame; identity must match.
    loop {
        let message = tokio::time::timeout(config.admit_timeout, stream.next())
            .await
            .map_err(|_| RealtimeClientError::AdmitTimeout)?
            .ok_or(RealtimeClientError::ClosedBeforeAdmit(1006))?
            .map_err(RealtimeClientError::Connect)?;
        match message {
            Message::Text(text) => match serde_json::from_str::<RealtimeServerFrame>(&text) {
                Ok(RealtimeServerFrame::Admitted {
                    session_id,
                    attempt_id,
                    worker_id,
                    node_id,
                    incarnation_id,
                    deployment_id,
                    model_generation: _,
                    output_audio,
                    bounds,
                }) => {
                    if session_id.as_str() != admit.session_id.as_str()
                        || attempt_id.as_str() != admit.attempt_id.as_str()
                    {
                        return Err(RealtimeClientError::Protocol(
                            "admitted frame does not echo the admit identity".into(),
                        ));
                    }
                    bounds.validate().map_err(|_| {
                        RealtimeClientError::Protocol(
                            "admitted bounds exceed the protocol caps".into(),
                        )
                    })?;
                    let stage_emits_audio =
                        matches!(admit.input, RealtimeStageInput::TextStream { .. });
                    if stage_emits_audio && output_audio.is_none() {
                        return Err(RealtimeClientError::Protocol(
                            "text-stream sessions must announce an output audio spec".into(),
                        ));
                    }
                    let admission = RealtimeAdmission {
                        session_id: session_id.to_string(),
                        attempt_id: attempt_id.to_string(),
                        worker_id: worker_id.to_string(),
                        node_id: node_id.to_string(),
                        incarnation_id: incarnation_id.to_string(),
                        deployment_id: deployment_id.to_string(),
                        bounds,
                        output_audio,
                    };
                    return Ok(DialedSession {
                        stream,
                        admission,
                        admitted_frame: text.to_string(),
                    });
                }
                Ok(
                    RealtimeServerFrame::Event { .. }
                    | RealtimeServerFrame::Pong
                    | RealtimeServerFrame::Unknown,
                ) => {
                    return Err(RealtimeClientError::Protocol(
                        "admitted frame must be the first server frame".into(),
                    ));
                }
                Err(error) => {
                    return Err(RealtimeClientError::Protocol(format!(
                        "admitted frame did not decode: {error}"
                    )));
                }
            },
            Message::Close(frame) => {
                return Err(RealtimeClientError::ClosedBeforeAdmit(
                    frame.map(|frame| u16::from(frame.code)).unwrap_or(1006),
                ));
            }
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Binary(_) | Message::Frame(_) => {
                return Err(RealtimeClientError::Protocol(
                    "worker sent a binary frame before admission".into(),
                ));
            }
        }
    }
}

/// The TLS connector for `wss` dials. Without private roots or a client
/// identity this is the dialer's default (public web roots); otherwise the
/// public roots plus the private CA roots, with the mTLS identity presented.
fn tls_connector(
    tls: &crate::WorkerClientTlsConfig,
) -> Result<Option<Connector>, RealtimeClientError> {
    if !tls.is_configured() {
        return Ok(None);
    }
    let invalid = |message| RealtimeClientError::InvalidConfiguration(message);
    let mut roots = rustls::RootCertStore {
        roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
    };
    for pem in &tls.private_ca_roots_pem {
        for der in
            crate::pem_certificates(pem).map_err(|_| invalid("private CA root is not valid PEM"))?
        {
            roots
                .add(rustls::pki_types::CertificateDer::from(der))
                .map_err(|_| invalid("private CA root is not a valid certificate"))?;
        }
    }
    let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|_| invalid("TLS client configuration could not be constructed"))?
    .with_root_certificates(roots);
    let config = match &tls.client_identity_pem {
        None => builder.with_no_client_auth(),
        Some(identity) => {
            let (certificates, key) = crate::pem_identity(identity)
                .map_err(|_| invalid("mTLS identity is not valid PEM"))?;
            builder
                .with_client_auth_cert(certificates, key)
                .map_err(|_| invalid("mTLS identity could not be used for TLS"))?
        }
    };
    Ok(Some(Connector::Rustls(Arc::new(config))))
}

async fn send_control_frame(
    stream: &mut WorkerSocket,
    send_timeout: Duration,
    frame: &RealtimeClientFrame,
) -> Result<(), RealtimeClientError> {
    let text = serde_json::to_string(frame)
        .map_err(|error| RealtimeClientError::Protocol(format!("control encode: {error}")))?;
    tokio::time::timeout(send_timeout, stream.send(Message::Text(text.into())))
        .await
        .map_err(|_| RealtimeClientError::SendTimeout)?
        .map_err(|error| RealtimeClientError::Protocol(format!("send failed: {error}")))
}

impl RealtimeSession {
    /// The negotiated session contract echoed by the worker.
    pub fn admission(&self) -> &RealtimeAdmission {
        &self.admission
    }

    /// True once exactly one terminal event has been observed.
    pub fn is_terminal(&self) -> bool {
        self.terminal_seen
    }

    /// Sends one PCM frame (little-endian signed 16-bit, at the sample rate
    /// declared in the admit audio spec). Frames are sequenced monotonically
    /// by the client; the admitted bounds are enforced before any bytes are
    /// written.
    pub async fn send_audio(&mut self, pcm_i16_le: &[u8]) -> Result<(), RealtimeClientError> {
        self.guard_open()?;
        if pcm_i16_le.len() > self.admission.bounds.max_frame_bytes {
            return Err(RealtimeClientError::AudioFrameTooLarge {
                actual: pcm_i16_le.len(),
                limit: self.admission.bounds.max_frame_bytes,
            });
        }
        let new_total = self.audio_bytes.saturating_add(pcm_i16_le.len() as u64);
        if new_total > self.admission.bounds.max_session_audio_bytes {
            return Err(RealtimeClientError::AudioBudgetExhausted {
                actual: new_total,
                limit: self.admission.bounds.max_session_audio_bytes,
            });
        }
        let sequence = self.next_audio_sequence;
        let frame = encode_realtime_audio_frame(sequence, false, pcm_i16_le).map_err(|error| {
            RealtimeClientError::Protocol(format!("audio frame encode rejected: {error}"))
        })?;
        tokio::time::timeout(
            self.config.send_timeout,
            self.stream.send(Message::Binary(frame.into())),
        )
        .await
        .map_err(|_| RealtimeClientError::SendTimeout)?
        .map_err(|error| RealtimeClientError::Protocol(format!("send failed: {error}")))?;
        self.next_audio_sequence = sequence.saturating_add(1);
        self.audio_bytes = new_total;
        Ok(())
    }

    /// Sends one bounded piece of the utterance text (TTS-stream stage).
    /// The worker accumulates pieces until [`RealtimeSession::finish`]; the
    /// per-frame protocol cap and the session text budget are enforced here
    /// before anything is written.
    pub async fn send_text(&mut self, text: &str) -> Result<(), RealtimeClientError> {
        self.guard_open()?;
        if !self.stage_emits_audio {
            return Err(RealtimeClientError::Protocol(
                "text input is not valid for audio-stream sessions".into(),
            ));
        }
        if text.len() > MAX_REALTIME_INPUT_TEXT_BYTES {
            return Err(RealtimeClientError::Protocol(format!(
                "input text of {} bytes exceeds the protocol bound {MAX_REALTIME_INPUT_TEXT_BYTES}",
                text.len()
            )));
        }
        let new_total = self.text_bytes.saturating_add(text.len() as u64);
        if new_total > self.admission.bounds.max_session_audio_bytes {
            return Err(RealtimeClientError::TextBudgetExhausted {
                actual: new_total,
                limit: self.admission.bounds.max_session_audio_bytes,
            });
        }
        self.send_control(&RealtimeClientFrame::Input {
            text: text.to_string(),
        })
        .await?;
        self.text_bytes = new_total;
        Ok(())
    }

    /// Signals end of input; the worker finalizes the stage and emits its
    /// final outputs followed by exactly one terminal event.
    pub async fn finish(&mut self) -> Result<(), RealtimeClientError> {
        self.guard_open()?;
        self.send_control(&RealtimeClientFrame::Finish).await
    }

    /// Requests cooperative cancellation, identical to the HTTP contract.
    pub async fn cancel(&mut self) -> Result<(), RealtimeClientError> {
        self.send_control(&RealtimeClientFrame::Cancel).await
    }

    /// Sends an application-level keepalive; the worker answers with a pong
    /// control frame, which [`RealtimeSession::next_event`] skips.
    pub async fn ping(&mut self) -> Result<(), RealtimeClientError> {
        self.send_control(&RealtimeClientFrame::Ping).await
    }

    /// Reads the next session event. Returns `Ok(None)` only after the
    /// terminal outcome has been observed and the worker closed the socket
    /// cleanly; every other end is an error. On stages that emit audio,
    /// interleaved audio frames are buffered (bounded) for
    /// [`RealtimeSession::next_audio`].
    pub async fn next_event(&mut self) -> Result<Option<InvocationEvent>, RealtimeClientError> {
        if let Some(event) = self.pending_events.pop_front() {
            return Ok(Some(event));
        }
        loop {
            if self.closed {
                return Ok(None);
            }
            let message = tokio::time::timeout(self.config.event_timeout, self.stream.next())
                .await
                .map_err(|_| RealtimeClientError::EventTimeout)?
                .ok_or_else(|| {
                    RealtimeClientError::Protocol("worker socket ended without a close".into())
                })?
                .map_err(|error| {
                    RealtimeClientError::Protocol(format!("session transport error: {error}"))
                })?;
            match message {
                Message::Text(text) => {
                    let frame: RealtimeServerFrame =
                        serde_json::from_str(&text).map_err(|error| {
                            RealtimeClientError::Protocol(format!(
                                "server frame did not decode: {error}"
                            ))
                        })?;
                    match frame {
                        RealtimeServerFrame::Event { event } => {
                            let event = self.validate_event(event)?;
                            return Ok(Some(event));
                        }
                        // A frame type from a newer minor is skipped.
                        RealtimeServerFrame::Pong | RealtimeServerFrame::Unknown => continue,
                        RealtimeServerFrame::Admitted { .. } => {
                            return Err(RealtimeClientError::Protocol(
                                "duplicate admitted frame".into(),
                            ));
                        }
                    }
                }
                Message::Binary(data) => {
                    if !self.stage_emits_audio {
                        return Err(RealtimeClientError::Protocol(
                            "worker sent an unexpected audio frame".into(),
                        ));
                    }
                    let chunk = self.decode_audio_frame(&data)?;
                    if self.pending_audio.len() >= self.admission.bounds.max_in_flight_frames {
                        return Err(RealtimeClientError::Protocol(
                            "too many audio frames buffered; drain them with next_audio".into(),
                        ));
                    }
                    self.pending_audio.push_back(chunk);
                }
                Message::Close(frame) => {
                    self.closed = true;
                    let code = frame.map(|frame| u16::from(frame.code)).unwrap_or(1000);
                    if !self.terminal_seen || code != 1000 {
                        return Err(RealtimeClientError::Closed(code));
                    }
                    return Ok(None);
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                Message::Frame(_) => {
                    return Err(RealtimeClientError::Protocol(
                        "worker sent an unexpected raw frame".into(),
                    ));
                }
            }
        }
    }

    /// Reads the next synthesized audio frame (TTS-stream stage). Events that
    /// arrive meanwhile are buffered (bounded) for
    /// [`RealtimeSession::next_event`]. Returns `Ok(None)` only after a clean
    /// close following the terminal outcome, mirroring [`Self::next_event`].
    pub async fn next_audio(&mut self) -> Result<Option<RealtimeAudioChunk>, RealtimeClientError> {
        if let Some(chunk) = self.pending_audio.pop_front() {
            return Ok(Some(chunk));
        }
        loop {
            if self.closed {
                return Ok(None);
            }
            let message = tokio::time::timeout(self.config.event_timeout, self.stream.next())
                .await
                .map_err(|_| RealtimeClientError::EventTimeout)?
                .ok_or_else(|| {
                    RealtimeClientError::Protocol("worker socket ended without a close".into())
                })?
                .map_err(|error| {
                    RealtimeClientError::Protocol(format!("session transport error: {error}"))
                })?;
            match message {
                Message::Binary(data) => {
                    if !self.stage_emits_audio {
                        return Err(RealtimeClientError::Protocol(
                            "worker sent an unexpected audio frame".into(),
                        ));
                    }
                    let chunk = self.decode_audio_frame(&data)?;
                    return Ok(Some(chunk));
                }
                Message::Text(text) => {
                    let frame: RealtimeServerFrame =
                        serde_json::from_str(&text).map_err(|error| {
                            RealtimeClientError::Protocol(format!(
                                "server frame did not decode: {error}"
                            ))
                        })?;
                    match frame {
                        RealtimeServerFrame::Event { event } => {
                            let event = self.validate_event(event)?;
                            if self.pending_events.len() >= MAX_PENDING_EVENTS {
                                return Err(RealtimeClientError::Protocol(
                                    "too many events buffered; drain them with next_event".into(),
                                ));
                            }
                            self.pending_events.push_back(event);
                        }
                        // A frame type from a newer minor is skipped.
                        RealtimeServerFrame::Pong | RealtimeServerFrame::Unknown => continue,
                        RealtimeServerFrame::Admitted { .. } => {
                            return Err(RealtimeClientError::Protocol(
                                "duplicate admitted frame".into(),
                            ));
                        }
                    }
                }
                Message::Close(frame) => {
                    self.closed = true;
                    let code = frame.map(|frame| u16::from(frame.code)).unwrap_or(1000);
                    if !self.terminal_seen || code != 1000 {
                        return Err(RealtimeClientError::Closed(code));
                    }
                    return Ok(None);
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                Message::Frame(_) => {
                    return Err(RealtimeClientError::Protocol(
                        "worker sent an unexpected raw frame".into(),
                    ));
                }
            }
        }
    }

    /// Decodes one worker-pushed IRTA frame against the announced bounds and
    /// the monotonic receive sequence.
    fn decode_audio_frame(
        &mut self,
        data: &[u8],
    ) -> Result<RealtimeAudioChunk, RealtimeClientError> {
        let (header, payload) = decode_realtime_audio_frame(data)
            .map_err(|error| RealtimeClientError::Protocol(format!("audio frame: {error}")))?;
        if payload.len() > self.admission.bounds.max_frame_bytes {
            return Err(RealtimeClientError::AudioFrameTooLarge {
                actual: payload.len(),
                limit: self.admission.bounds.max_frame_bytes,
            });
        }
        if self
            .last_received_audio_sequence
            .is_some_and(|last| header.sequence <= last)
        {
            return Err(RealtimeClientError::Protocol(format!(
                "audio frame sequence {} did not advance",
                header.sequence
            )));
        }
        self.last_received_audio_sequence = Some(header.sequence);
        Ok(RealtimeAudioChunk {
            sequence: header.sequence,
            is_final: header.is_final,
            payload: payload.to_vec(),
        })
    }

    async fn send_control(
        &mut self,
        frame: &RealtimeClientFrame,
    ) -> Result<(), RealtimeClientError> {
        send_control_frame(&mut self.stream, self.config.send_timeout, frame).await
    }

    fn guard_open(&self) -> Result<(), RealtimeClientError> {
        if self.terminal_seen || self.closed {
            return Err(RealtimeClientError::Protocol(
                "session already produced its terminal outcome".into(),
            ));
        }
        Ok(())
    }

    /// One Accepted first, strictly contiguous sequences, matching identity,
    /// exactly one terminal outcome, nothing after it.
    fn validate_event(
        &mut self,
        event: InvocationEvent,
    ) -> Result<InvocationEvent, RealtimeClientError> {
        if event.schema_version.major != PROTOCOL_V1.major {
            return Err(RealtimeClientError::Protocol(format!(
                "event schema major {} is not supported",
                event.schema_version.major
            )));
        }
        if event.request_id.as_str() != self.expected_request_id {
            return Err(RealtimeClientError::Protocol(
                "event request id does not match the admit".into(),
            ));
        }
        if event.attempt_id.as_str() != self.admission.attempt_id {
            return Err(RealtimeClientError::Protocol(
                "event attempt id does not match the admit".into(),
            ));
        }
        if self.terminal_seen {
            return Err(RealtimeClientError::Protocol(
                "event arrived after the terminal outcome".into(),
            ));
        }
        let expected_sequence = self.last_event_sequence.map_or(0, |last| last + 1);
        if event.sequence != expected_sequence {
            return Err(RealtimeClientError::Protocol(format!(
                "event sequence {} is not contiguous (expected {expected_sequence})",
                event.sequence
            )));
        }
        match &event.event {
            InvocationEventKind::Accepted { .. } => {
                if self.accepted_seen {
                    return Err(RealtimeClientError::Protocol(
                        "duplicate accepted event".into(),
                    ));
                }
                if event.sequence != 0 {
                    return Err(RealtimeClientError::Protocol(
                        "accepted event must be sequence 0".into(),
                    ));
                }
                self.accepted_seen = true;
            }
            InvocationEventKind::Completed { .. }
            | InvocationEventKind::Error { .. }
            | InvocationEventKind::Cancelled { .. } => {
                if !self.accepted_seen {
                    return Err(RealtimeClientError::Protocol(
                        "terminal event arrived before accepted".into(),
                    ));
                }
                self.terminal_seen = true;
            }
            InvocationEventKind::TextDelta { .. }
            | InvocationEventKind::Usage { .. }
            | InvocationEventKind::Unknown => {
                if !self.accepted_seen {
                    return Err(RealtimeClientError::Protocol(
                        "stream event arrived before accepted".into(),
                    ));
                }
            }
        }
        self.last_event_sequence = Some(event.sequence);
        Ok(event)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn credentials() -> ServiceCredentials {
        ServiceCredentials {
            credential_id: izwi_serving_protocol::CredentialId::new("credential-1").unwrap(),
            bearer_token: izwi_serving_protocol::ServiceBearerToken::new("session-secret").unwrap(),
        }
    }

    fn admission() -> RealtimeSessionAdmit {
        use std::collections::BTreeSet;
        let id = |value: &'static str| izwi_serving_protocol::AttemptId::new(value).unwrap();
        RealtimeSessionAdmit {
            schema_version: PROTOCOL_V1,
            session_id: izwi_serving_protocol::SessionId::new("session-1").unwrap(),
            request_id: izwi_serving_protocol::RequestId::new("request-1").unwrap(),
            attempt_id: id("attempt-1"),
            expected_worker_incarnation: izwi_serving_protocol::IncarnationId::new("inc-1")
                .unwrap(),
            deployment_id: izwi_serving_protocol::DeploymentId::new("dep-1").unwrap(),
            expected_model_generation: izwi_serving_protocol::ModelGeneration::new(1).unwrap(),
            caller: izwi_serving_protocol::GatewayAttestedCallerContext {
                tenant_id: izwi_serving_protocol::TenantId::new("tenant-1").unwrap(),
                caller_id: izwi_serving_protocol::CallerId::new("caller-1").unwrap(),
                policy_revision: izwi_serving_protocol::PolicyRevision::new("policy-1").unwrap(),
                permitted_actions: BTreeSet::from([izwi_serving_protocol::PermittedAction::Invoke]),
                allowed_data_regions: vec!["local".into()],
            },
            task: izwi_serving_protocol::TaskKind::SpeechToText,
            service_class: izwi_serving_protocol::ServiceClass::Realtime,
            remaining_time_ms: 5_000,
            input: izwi_serving_protocol::RealtimeStageInput::AudioStream {
                spec: izwi_serving_protocol::RealtimeAudioSpec {
                    codec: izwi_serving_protocol::RealtimeAudioCodec::PcmI16Le,
                    sample_rate: 16_000,
                    channels: 1,
                },
                language: None,
            },
        }
    }

    #[test]
    fn realtime_connect_rejects_plaintext_and_unsupported_admit_before_dialing() {
        let config = RealtimeClientConfig::default();
        for endpoint in [
            "http://localhost:9470",
            "http://192.0.2.1:9470",
            "http://user:password@127.0.0.1:9470",
            "not a url",
        ] {
            let error = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(connect(
                    endpoint,
                    &credentials(),
                    admission(),
                    config.clone(),
                ))
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    RealtimeClientError::InvalidConfiguration(_)
                        | RealtimeClientError::InvalidEndpoint(_)
                ),
                "unexpected error for {endpoint}: {error}"
            );
            assert!(error.proves_session_unaccepted());
        }
    }

    // A test CA, a localhost/127.0.0.1 server certificate and a client
    // (mTLS) certificate it signed, all valid until 2126.
    const TEST_CA_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBqDCCAU2gAwIBAgIUTSeQRvVl8fjJWjOAr5v+z9ErlUIwCgYIKoZIzj0EAwIw
IDEeMBwGA1UEAwwVaXp3aS1yZWFsdGltZS10ZXN0LWNhMCAXDTI2MTAxNjE2MDQz
NVoYDzIxMjYwOTIyMTYwNDM1WjAgMR4wHAYDVQQDDBVpendpLXJlYWx0aW1lLXRl
c3QtY2EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAASDRarRenz0/t4mgONOgdwR
cGehlkVMpZ+b4S3GVcNQEM/wX1VmjNqaiZf8F8S6c5xc4hyqu386taS/NTAYg20s
o2MwYTAdBgNVHQ4EFgQUZYkQdE5VD/eWOILaOvxYqwWJynQwHwYDVR0jBBgwFoAU
ZYkQdE5VD/eWOILaOvxYqwWJynQwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8E
BAMCAgQwCgYIKoZIzj0EAwIDSQAwRgIhAPKi8OxP4WuBMQNh5LlHYo0yyB3EWKK6
hmsSvcKYLtGpAiEA3S/fw3eOxwH0d9rDFB8IgrEeF5Y7XFcXhTZ7IlmSRtU=
-----END CERTIFICATE-----
";
    const TEST_SERVER_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBzDCCAXGgAwIBAgIUV4Od/WnkLvfvSzndwkwZ+2KKSBcwCgYIKoZIzj0EAwIw
IDEeMBwGA1UEAwwVaXp3aS1yZWFsdGltZS10ZXN0LWNhMCAXDTI2MTAxNjE2MDQz
NVoYDzIxMjYwOTIyMTYwNDM1WjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwWTATBgcq
hkjOPQIBBggqhkjOPQMBBwNCAAQ41lnWt9tcH8FdAmpa393twxE3Dnl0piN0m9XR
uOOyXqukdeoOebkOONMlnQDTu5wRHdJ7zLoDyCVoZpH/oRYpo4GSMIGPMBoGA1Ud
EQQTMBGCCWxvY2FsaG9zdIcEfwAAATAMBgNVHRMBAf8EAjAAMA4GA1UdDwEB/wQE
AwIHgDATBgNVHSUEDDAKBggrBgEFBQcDATAdBgNVHQ4EFgQUs9/z0fzMQ/xmyO6B
IHCP+1jusCMwHwYDVR0jBBgwFoAUZYkQdE5VD/eWOILaOvxYqwWJynQwCgYIKoZI
zj0EAwIDSQAwRgIhALJfKxGxbeAR1Wzvd08Xqt1QmZYksv0WxsIySlYqQh/IAiEA
/yRaLXt/VZEuRmi0eDvz7lBZnDXDv2xLZH6zJXwt/YE=
-----END CERTIFICATE-----
";
    const TEST_SERVER_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgqMNQEzVw4fFfbGO4
OfOjjdS8jdcFVyE+N+9oSwUGoA6hRANCAAQ41lnWt9tcH8FdAmpa393twxE3Dnl0
piN0m9XRuOOyXqukdeoOebkOONMlnQDTu5wRHdJ7zLoDyCVoZpH/oRYp
-----END PRIVATE KEY-----
";
    const TEST_CLIENT_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----
MIIBsDCCAVagAwIBAgIUV4Od/WnkLvfvSzndwkwZ+2KKSBgwCgYIKoZIzj0EAwIw
IDEeMBwGA1UEAwwVaXp3aS1yZWFsdGltZS10ZXN0LWNhMCAXDTI2MTAxNjE2MDQz
NVoYDzIxMjYwOTIyMTYwNDM1WjAXMRUwEwYDVQQDDAxpendpLWdhdGV3YXkwWTAT
BgcqhkjOPQIBBggqhkjOPQMBBwNCAAQc/MGdk7Hw/L8SwQ310bfgOIk1zLk34+my
u+6qqH+fsAnu9WiQRDr+hzrC+Ae6lIKjktspbkiNQ0ad4EAP01bNo3UwczAMBgNV
HRMBAf8EAjAAMA4GA1UdDwEB/wQEAwIHgDATBgNVHSUEDDAKBggrBgEFBQcDAjAd
BgNVHQ4EFgQUjYtq01QMeMM3cFBPcqOd1Sjh0AMwHwYDVR0jBBgwFoAUZYkQdE5V
D/eWOILaOvxYqwWJynQwCgYIKoZIzj0EAwIDSAAwRQIhAI2UUfKhGcsl90WdVHRI
owpIUlPejzCD78TLXCxcAw0hAiBJKSGM8usNlTQs/zMYMGr6u2t75wZi4fpbb4BY
RDtcJw==
-----END CERTIFICATE-----
";
    const TEST_CLIENT_KEY_PEM: &str = "-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgpogiAPFDRSvtKgDe
7vaqz2ON6O1LC6NEueNWcEwMzpOhRANCAAQc/MGdk7Hw/L8SwQ310bfgOIk1zLk3
4+myu+6qqH+fsAnu9WiQRDr+hzrC+Ae6lIKjktspbkiNQ0ad4EAP01bN
-----END PRIVATE KEY-----
";

    /// A TLS worker socket that requires a client certificate from the test
    /// CA and echoes every admit. Returns its port.
    // tungstenite's handshake callback signature returns its large error.
    #[allow(clippy::result_large_err)]
    async fn mtls_worker() -> u16 {
        use tokio_rustls::rustls::pki_types::{CertificateDer, PrivateKeyDer};

        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut roots = rustls::RootCertStore::empty();
        for der in crate::pem_certificates(TEST_CA_PEM.as_bytes()).unwrap() {
            roots.add(CertificateDer::from(der)).unwrap();
        }
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(roots),
            Arc::clone(&provider),
        )
        .build()
        .unwrap();
        let certificates = crate::pem_certificates(TEST_SERVER_CERT_PEM.as_bytes())
            .unwrap()
            .into_iter()
            .map(CertificateDer::from)
            .collect();
        let (_, key): (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) =
            crate::pem_identity(TEST_SERVER_KEY_PEM.as_bytes()).unwrap();
        let config = rustls::ServerConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, key)
            .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let Ok(tls) = acceptor.accept(stream).await else {
                    continue;
                };
                let mut socket = tokio_tungstenite::accept_hdr_async(
                    tls,
                    |_: &tokio_tungstenite::tungstenite::handshake::server::Request,
                     mut response: tokio_tungstenite::tungstenite::handshake::server::Response| {
                        response.headers_mut().insert(
                            "sec-websocket-protocol",
                            HeaderValue::from_static(REALTIME_SUBPROTOCOL),
                        );
                        Ok(response)
                    },
                )
                .await
                .unwrap();
                let Some(Ok(Message::Text(text))) = socket.next().await else {
                    continue;
                };
                let Ok(RealtimeClientFrame::Admit { admit }) = serde_json::from_str(&text) else {
                    continue;
                };
                let admitted = RealtimeServerFrame::Admitted {
                    session_id: admit.session_id,
                    attempt_id: admit.attempt_id,
                    worker_id: izwi_serving_protocol::WorkerId::new("worker-1").unwrap(),
                    node_id: izwi_serving_protocol::NodeId::new("node-1").unwrap(),
                    incarnation_id: admit.expected_worker_incarnation,
                    deployment_id: admit.deployment_id,
                    model_generation: admit.expected_model_generation,
                    output_audio: None,
                    bounds: RealtimeSessionBounds {
                        max_frame_bytes: 4096,
                        max_in_flight_frames: 4,
                        max_session_audio_bytes: 65_536,
                    },
                };
                let _ = socket
                    .send(Message::Text(
                        serde_json::to_string(&admitted).unwrap().into(),
                    ))
                    .await;
            }
        });
        port
    }

    #[tokio::test]
    async fn dial_verifies_the_private_ca_and_presents_the_mtls_identity() {
        let port = mtls_worker().await;
        let endpoint = format!("https://localhost:{port}/");
        let tls = |identity: bool| {
            crate::WorkerClientTlsConfig::from_pem(
                vec![TEST_CA_PEM.as_bytes().to_vec()],
                identity.then(|| TEST_CLIENT_CERT_PEM.as_bytes().to_vec()),
                identity.then(|| TEST_CLIENT_KEY_PEM.as_bytes().to_vec()),
            )
            .unwrap()
        };
        let dial_with = |tls| {
            let endpoint = endpoint.clone();
            async move {
                let config = RealtimeClientConfig {
                    tls,
                    ..RealtimeClientConfig::default()
                };
                dial(&endpoint, &credentials(), &admission(), None, &config).await
            }
        };

        let dialed = dial_with(tls(true)).await.unwrap();
        assert_eq!(dialed.admission.session_id, "session-1");
        assert!(dialed.admitted_frame.contains("\"admitted\""));

        // Without the private CA the worker certificate is untrusted, and
        // without the identity the worker refuses the handshake.
        for config in [crate::WorkerClientTlsConfig::default(), tls(false)] {
            let error = dial_with(config).await.unwrap_err();
            assert!(error.proves_session_unaccepted(), "{error}");
        }
    }

    #[test]
    fn realtime_config_defaults_are_bounded() {
        let config = RealtimeClientConfig::default();
        assert!(!config.connect_timeout.is_zero());
        assert!(!config.admit_timeout.is_zero());
        assert!(!config.event_timeout.is_zero());
        assert!(!config.send_timeout.is_zero());
    }
}
