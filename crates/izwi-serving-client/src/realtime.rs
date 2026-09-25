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
//! Dropping a session without a terminal outcome leaves the worker session
//! running until its own deadline; the owner is responsible for an explicit
//! HTTP `cancel_attempt` in that case, exactly as for an uncertain HTTP
//! invocation.

use futures::{SinkExt, StreamExt};
use izwi_serving_protocol::{
    encode_realtime_audio_frame, InvocationEvent, InvocationEventKind, RealtimeClientFrame,
    RealtimeServerFrame, RealtimeSessionAdmit, RealtimeSessionBounds, ServiceCredentials,
    PROTOCOL_V1, REALTIME_SUBPROTOCOL, REALTIME_WS_PATH, SERVICE_AUTHORIZATION_HEADER,
    SERVICE_AUTH_SCHEME, SERVICE_CREDENTIAL_ID_HEADER,
};
use std::time::Duration;
use tokio_tungstenite::tungstenite::{
    client::IntoClientRequest, http::HeaderValue, Error as WsError, Message,
};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
}

impl Default for RealtimeClientConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            admit_timeout: Duration::from_secs(10),
            event_timeout: Duration::from_secs(30),
            send_timeout: Duration::from_secs(5),
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
}

/// One admitted realtime session over a worker WebSocket.
pub struct RealtimeSession {
    stream: WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    admission: RealtimeAdmission,
    expected_request_id: String,
    config: RealtimeClientConfig,
    accepted_seen: bool,
    terminal_seen: bool,
    closed: bool,
    last_event_sequence: Option<u64>,
    next_audio_sequence: u32,
    audio_bytes: u64,
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

/// Opens the session: upgrade, subprotocol negotiation, and the mandatory
/// first admit frame, resolved only once the worker's `Admitted` echo
/// arrives and matches the admit identity.
pub async fn connect(
    endpoint: &str,
    credentials: &ServiceCredentials,
    admit: RealtimeSessionAdmit,
    config: RealtimeClientConfig,
) -> Result<RealtimeSession, RealtimeClientError> {
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
    let ws_url = base
        .join(REALTIME_WS_PATH.trim_start_matches('/'))
        .map_err(|error| RealtimeClientError::InvalidEndpoint(error.to_string()))?
        .to_string();
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

    let handshake = tokio_tungstenite::connect_async(request);
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
    let admission = loop {
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
                    bounds,
                    ..
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
                    break RealtimeAdmission {
                        session_id: session_id.to_string(),
                        attempt_id: attempt_id.to_string(),
                        worker_id: worker_id.to_string(),
                        node_id: node_id.to_string(),
                        incarnation_id: incarnation_id.to_string(),
                        deployment_id: deployment_id.to_string(),
                        bounds,
                    };
                }
                Ok(RealtimeServerFrame::Event { .. } | RealtimeServerFrame::Pong) => {
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
    };

    Ok(RealtimeSession {
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
    })
}

async fn send_control_frame(
    stream: &mut WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
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
    /// cleanly; every other end is an error.
    pub async fn next_event(&mut self) -> Result<Option<InvocationEvent>, RealtimeClientError> {
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
                        RealtimeServerFrame::Event { event } => self.validate_event(event)?,
                        RealtimeServerFrame::Pong => continue,
                        RealtimeServerFrame::Admitted { .. } => {
                            return Err(RealtimeClientError::Protocol(
                                "duplicate admitted frame".into(),
                            ));
                        }
                    }
                }
                Message::Binary(_) => {
                    return Err(RealtimeClientError::Protocol(
                        "worker sent an unexpected audio frame".into(),
                    ));
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
    fn validate_event(&mut self, event: InvocationEvent) -> Result<(), RealtimeClientError> {
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
            InvocationEventKind::TextDelta { .. } | InvocationEventKind::Usage { .. } => {
                if !self.accepted_seen {
                    return Err(RealtimeClientError::Protocol(
                        "stream event arrived before accepted".into(),
                    ));
                }
            }
        }
        self.last_event_sequence = Some(event.sequence);
        Ok(())
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

    #[test]
    fn realtime_config_defaults_are_bounded() {
        let config = RealtimeClientConfig::default();
        assert!(!config.connect_timeout.is_zero());
        assert!(!config.admit_timeout.is_zero());
        assert!(!config.event_timeout.is_zero());
        assert!(!config.send_timeout.is_zero());
    }
}
