//! Versioned realtime WebSocket subprotocol between gateways and workers.
//!
//! The realtime surface multiplexes one bounded JSON control channel (text
//! frames) with raw binary audio frames on a single WebSocket. One session
//! binds exactly one stage (`TaskKind::SpeechToText` or
//! `TaskKind::TextToSpeech`) to exactly one worker; a multi-stage voice
//! workflow is composed gateway-side from one session per stage.
//!
//! Session admission reuses the HTTP invocation identity model: the admit
//! carries gateway-minted `request_id`/`attempt_id`, so the worker registers
//! the session in the same attempt table and `query_attempt`/`cancel_attempt`
//! over HTTP behave identically for realtime sessions. After the worker's
//! `admitted` frame, every event frame is a typed [`InvocationEvent`], so the
//! terminal-event vocabulary and one-terminal-outcome rule are shared with
//! the NDJSON path.

use crate::identity::*;
use crate::types::{
    GatewayAttestedCallerContext, InvocationEvent, SchemaVersion, ServiceClass, TaskKind,
    MAX_CALLER_REGIONS, MAX_REMAINING_TIME_MS,
};
use crate::PROTOCOL_V1;
use serde::{Deserialize, Serialize};

/// The only realtime subprotocol implemented by this crate.
pub const REALTIME_SUBPROTOCOL: &str = "izwi-realtime-v1";
/// Worker-side WebSocket upgrade path for realtime sessions.
pub const REALTIME_WS_PATH: &str = "/internal/v1/realtime";

/// Hard per-frame payload cap for binary audio frames.
pub const MAX_REALTIME_AUDIO_FRAME_BYTES: usize = 512 * 1024;
/// Hard cap on input text delivered through a control frame (TTS stage).
pub const MAX_REALTIME_INPUT_TEXT_BYTES: usize = 256 * 1024;
/// Hard cap on audio frames buffered in flight for one session.
pub const MAX_REALTIME_IN_FLIGHT_FRAMES: usize = 64;
/// Hard cap on total audio bytes admitted per realtime session.
pub const MAX_REALTIME_SESSION_AUDIO_BYTES: u64 = 1024 * 1024 * 1024;
/// Bounded language tag length on an audio-stream admission.
pub const MAX_REALTIME_LANGUAGE_BYTES: usize = 128;

/// Binary audio frame magic (`IZWI RealTime Audio`).
pub const REALTIME_AUDIO_MAGIC: [u8; 4] = *b"IRTA";
/// Binary audio frame layout version.
pub const REALTIME_AUDIO_VERSION: u8 = 1;
/// Byte length of the fixed binary audio frame header.
pub const REALTIME_AUDIO_HEADER_BYTES: usize = 16;
/// Bit 0 of the flags byte marks the final frame of a direction.
pub const REALTIME_AUDIO_FLAG_FINAL: u8 = 0b0000_0001;

/// Audio encodings understood by `izwi-realtime-v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeAudioCodec {
    /// Little-endian signed 16-bit PCM, interleaved if channels > 1.
    PcmI16Le,
}

/// Declared audio format for one direction of a realtime session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeAudioSpec {
    pub codec: RealtimeAudioCodec,
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeAudioSpecError;

impl RealtimeAudioSpec {
    /// Validates the declared format against the v1 envelope's supported
    /// range: mono PCM16 between 8 kHz and 192 kHz, matching the local
    /// realtime voice input contract.
    pub fn validate(&self) -> Result<(), RealtimeAudioSpecError> {
        if self.codec != RealtimeAudioCodec::PcmI16Le
            || self.channels != 1
            || !(8_000..=192_000).contains(&self.sample_rate)
        {
            return Err(RealtimeAudioSpecError);
        }
        Ok(())
    }

    pub const fn bytes_per_sample(&self) -> usize {
        2 * self.channels as usize
    }
}

/// Negotiated per-session transport bounds, echoed by the worker in
/// [`RealtimeServerFrame::Admitted`]. Values are always within the hard caps
/// above; a worker may negotiate lower values than its configured maxima but
/// never higher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeSessionBounds {
    pub max_frame_bytes: usize,
    pub max_in_flight_frames: usize,
    pub max_session_audio_bytes: u64,
}

impl RealtimeSessionBounds {
    pub fn validate(&self) -> Result<(), RealtimeContractError> {
        if self.max_frame_bytes == 0
            || self.max_frame_bytes > MAX_REALTIME_AUDIO_FRAME_BYTES
            || self.max_in_flight_frames == 0
            || self.max_in_flight_frames > MAX_REALTIME_IN_FLIGHT_FRAMES
            || self.max_session_audio_bytes == 0
            || self.max_session_audio_bytes > MAX_REALTIME_SESSION_AUDIO_BYTES
        {
            return Err(RealtimeContractError::InvalidBounds);
        }
        Ok(())
    }
}

/// Stage-specific admission input for a realtime session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RealtimeStageInput {
    /// Client pushes audio into the worker (ASR-stream stage).
    AudioStream {
        spec: RealtimeAudioSpec,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    /// Client pushes bounded text into the worker (TTS-stream stage).
    TextStream,
}

impl RealtimeStageInput {
    pub const fn task(&self) -> TaskKind {
        match self {
            Self::AudioStream { .. } => TaskKind::SpeechToText,
            Self::TextStream => TaskKind::TextToSpeech,
        }
    }
}

/// First client frame on a realtime session: admission through the same
/// fencing identity model as an HTTP `InvocationRequest`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeSessionAdmit {
    pub schema_version: SchemaVersion,
    pub session_id: SessionId,
    pub request_id: RequestId,
    pub attempt_id: AttemptId,
    pub expected_worker_incarnation: IncarnationId,
    pub deployment_id: DeploymentId,
    pub expected_model_generation: ModelGeneration,
    pub caller: GatewayAttestedCallerContext,
    pub task: TaskKind,
    pub service_class: ServiceClass,
    /// Remaining end-to-end session budget when sent by the gateway.
    pub remaining_time_ms: u64,
    pub input: RealtimeStageInput,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RealtimeContractError {
    #[error("unsupported protocol version {actual_major}.{actual_minor}; expected major {expected_major}")]
    UnsupportedVersion {
        actual_major: u16,
        actual_minor: u16,
        expected_major: u16,
    },
    #[error("realtime sessions accept only speech_to_text and text_to_speech stages")]
    UnsupportedTask,
    #[error("realtime sessions require the realtime service class")]
    UnsupportedServiceClass,
    #[error("admitted task {declared:?} does not match the typed stage input task {actual:?}")]
    TaskMismatch {
        declared: TaskKind,
        actual: TaskKind,
    },
    #[error("session budget must be non-zero and within the protocol maximum")]
    InvalidExecutionBudget,
    #[error("caller context contains too many region constraints")]
    TooManyCallerRegions,
    #[error("audio spec is outside the v1 supported range")]
    InvalidAudioSpec,
    #[error("language tag exceeds {MAX_REALTIME_LANGUAGE_BYTES} bytes")]
    InvalidLanguageTag,
    #[error("input text exceeds {MAX_REALTIME_INPUT_TEXT_BYTES} bytes")]
    InvalidInputText,
    #[error("session bounds are outside the hard protocol caps")]
    InvalidBounds,
    #[error("binary audio frame is malformed: {0}")]
    MalformedAudioFrame(&'static str),
    #[error("binary audio frame payload exceeds the hard frame cap")]
    AudioFrameTooLarge,
}

impl RealtimeSessionAdmit {
    pub fn validate(&self) -> Result<(), RealtimeContractError> {
        if self.schema_version.major != PROTOCOL_V1.major {
            return Err(RealtimeContractError::UnsupportedVersion {
                actual_major: self.schema_version.major,
                actual_minor: self.schema_version.minor,
                expected_major: PROTOCOL_V1.major,
            });
        }
        let input_task = self.input.task();
        if self.task == TaskKind::Chat || input_task == TaskKind::Chat {
            return Err(RealtimeContractError::UnsupportedTask);
        }
        if self.task != input_task {
            return Err(RealtimeContractError::TaskMismatch {
                declared: self.task,
                actual: input_task,
            });
        }
        if self.service_class != ServiceClass::Realtime {
            return Err(RealtimeContractError::UnsupportedServiceClass);
        }
        if self.remaining_time_ms == 0 || self.remaining_time_ms > MAX_REMAINING_TIME_MS {
            return Err(RealtimeContractError::InvalidExecutionBudget);
        }
        if self.caller.allowed_data_regions.len() > MAX_CALLER_REGIONS {
            return Err(RealtimeContractError::TooManyCallerRegions);
        }
        match &self.input {
            RealtimeStageInput::AudioStream { spec, language } => {
                spec.validate()
                    .map_err(|_| RealtimeContractError::InvalidAudioSpec)?;
                if language
                    .as_ref()
                    .is_some_and(|tag| tag.len() > MAX_REALTIME_LANGUAGE_BYTES)
                {
                    return Err(RealtimeContractError::InvalidLanguageTag);
                }
            }
            RealtimeStageInput::TextStream => {}
        }
        Ok(())
    }
}

/// Client-to-worker JSON control frames. The first frame on a session must be
/// [`RealtimeClientFrame::Admit`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeClientFrame {
    Admit {
        admit: Box<RealtimeSessionAdmit>,
    },
    /// End of client input: the worker finalizes the stage and emits its
    /// final outputs followed by exactly one terminal event.
    Finish,
    /// Cooperative cancellation, identical to the HTTP cancel contract.
    Cancel,
    Ping,
    /// Bounded text input (TTS-stream stage).
    Input {
        text: String,
    },
}

/// Worker-to-client JSON control frames. `Admitted` is always the first
/// frame; every later event frame wraps a typed [`InvocationEvent`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RealtimeServerFrame {
    Admitted {
        session_id: SessionId,
        attempt_id: AttemptId,
        worker_id: WorkerId,
        node_id: NodeId,
        incarnation_id: IncarnationId,
        deployment_id: DeploymentId,
        model_generation: ModelGeneration,
        /// Output audio spec for stages that emit audio (TTS-stream).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output_audio: Option<RealtimeAudioSpec>,
        bounds: RealtimeSessionBounds,
    },
    Pong,
    Event {
        event: InvocationEvent,
    },
}

/// Fixed-layout binary audio frame: 16-byte header followed by the codec
/// payload declared in the session's audio spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealtimeAudioFrameHeader {
    pub is_final: bool,
    pub sequence: u32,
    pub payload_len: usize,
}

/// Encodes one binary audio frame. Payload length is checked against the
/// hard frame cap before any allocation.
pub fn encode_realtime_audio_frame(
    sequence: u32,
    is_final: bool,
    payload: &[u8],
) -> Result<Vec<u8>, RealtimeContractError> {
    if payload.len() > MAX_REALTIME_AUDIO_FRAME_BYTES {
        return Err(RealtimeContractError::AudioFrameTooLarge);
    }
    let mut frame = Vec::with_capacity(REALTIME_AUDIO_HEADER_BYTES + payload.len());
    frame.extend_from_slice(&REALTIME_AUDIO_MAGIC);
    frame.push(REALTIME_AUDIO_VERSION);
    frame.push(if is_final {
        REALTIME_AUDIO_FLAG_FINAL
    } else {
        0
    });
    frame.extend_from_slice(&0_u16.to_le_bytes());
    frame.extend_from_slice(&sequence.to_le_bytes());
    frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    frame.extend_from_slice(payload);
    debug_assert_eq!(frame.len(), REALTIME_AUDIO_HEADER_BYTES + payload.len());
    Ok(frame)
}

/// Decodes one binary audio frame, returning its header and payload slice.
pub fn decode_realtime_audio_frame(
    frame: &[u8],
) -> Result<(RealtimeAudioFrameHeader, &[u8]), RealtimeContractError> {
    if frame.len() < REALTIME_AUDIO_HEADER_BYTES {
        return Err(RealtimeContractError::MalformedAudioFrame(
            "truncated header",
        ));
    }
    if frame[0..4] != REALTIME_AUDIO_MAGIC {
        return Err(RealtimeContractError::MalformedAudioFrame("bad magic"));
    }
    if frame[4] != REALTIME_AUDIO_VERSION {
        return Err(RealtimeContractError::MalformedAudioFrame(
            "unsupported version",
        ));
    }
    let payload_len = u32::from_le_bytes(frame[12..16].try_into().expect("fixed header")) as usize;
    if frame.len() - REALTIME_AUDIO_HEADER_BYTES != payload_len {
        return Err(RealtimeContractError::MalformedAudioFrame(
            "payload length does not match frame length",
        ));
    }
    if payload_len > MAX_REALTIME_AUDIO_FRAME_BYTES {
        return Err(RealtimeContractError::AudioFrameTooLarge);
    }
    let header = RealtimeAudioFrameHeader {
        is_final: frame[5] & REALTIME_AUDIO_FLAG_FINAL != 0,
        sequence: u32::from_le_bytes(frame[8..12].try_into().expect("fixed header")),
        payload_len,
    };
    Ok((header, &frame[REALTIME_AUDIO_HEADER_BYTES..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::ServiceBearerToken;
    use crate::types::{
        AttemptState, CancellationBehavior, ChatInput, ChatMessage, ChatParameters, ChatRole,
        InvocationEventKind, InvocationInput, OutputFormat, PermittedAction, WorkerFeature,
    };
    use std::collections::BTreeSet;

    fn admit(task: TaskKind) -> RealtimeSessionAdmit {
        RealtimeSessionAdmit {
            schema_version: PROTOCOL_V1,
            session_id: SessionId::new("sess-1").expect("session id"),
            request_id: RequestId::new("req-1").expect("request id"),
            attempt_id: AttemptId::new("att-1").expect("attempt id"),
            expected_worker_incarnation: IncarnationId::new("inc-1").expect("incarnation"),
            deployment_id: DeploymentId::new("dep-1").expect("deployment"),
            expected_model_generation: ModelGeneration::new(7).expect("generation"),
            caller: GatewayAttestedCallerContext {
                tenant_id: TenantId::new("tenant-a").expect("tenant"),
                caller_id: CallerId::new("caller-1").expect("caller"),
                policy_revision: PolicyRevision::new("policy-1").expect("policy"),
                permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                allowed_data_regions: Vec::new(),
            },
            task,
            service_class: ServiceClass::Realtime,
            remaining_time_ms: 60_000,
            input: RealtimeStageInput::AudioStream {
                spec: RealtimeAudioSpec {
                    codec: RealtimeAudioCodec::PcmI16Le,
                    sample_rate: 16_000,
                    channels: 1,
                },
                language: Some("en".into()),
            },
        }
    }

    #[test]
    fn admit_validates_audio_stream_stage() {
        let admit = admit(TaskKind::SpeechToText);
        admit.validate().expect("valid admit");
    }

    #[test]
    fn admit_rejects_chat_stage() {
        let mut admit = admit(TaskKind::Chat);
        admit.input = RealtimeStageInput::TextStream;
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::UnsupportedTask)
        ));
    }

    #[test]
    fn admit_rejects_task_mismatch() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.task = TaskKind::TextToSpeech;
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::TaskMismatch { .. })
        ));
    }

    #[test]
    fn admit_rejects_non_realtime_service_class() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.service_class = ServiceClass::Interactive;
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::UnsupportedServiceClass)
        ));
    }

    #[test]
    fn admit_rejects_zero_budget() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.remaining_time_ms = 0;
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::InvalidExecutionBudget)
        ));
    }

    #[test]
    fn admit_rejects_unsupported_major_version() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.schema_version = SchemaVersion::new(2, 0);
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn admit_rejects_out_of_range_audio_spec() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.input = RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: 4_000,
                channels: 1,
            },
            language: None,
        };
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::InvalidAudioSpec)
        ));
    }

    #[test]
    fn admit_rejects_oversized_language_tag() {
        let mut admit = admit(TaskKind::SpeechToText);
        admit.input = RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: 16_000,
                channels: 1,
            },
            language: Some("x".repeat(MAX_REALTIME_LANGUAGE_BYTES + 1)),
        };
        assert!(matches!(
            admit.validate(),
            Err(RealtimeContractError::InvalidLanguageTag)
        ));
    }

    #[test]
    fn client_frames_round_trip_serde() {
        let admit = admit(TaskKind::SpeechToText);
        let frames = [
            RealtimeClientFrame::Admit {
                admit: Box::new(admit.clone()),
            },
            RealtimeClientFrame::Finish,
            RealtimeClientFrame::Cancel,
            RealtimeClientFrame::Ping,
            RealtimeClientFrame::Input {
                text: "hello".into(),
            },
        ];
        for frame in frames {
            let json = serde_json::to_string(&frame).expect("serialize");
            let decoded: RealtimeClientFrame = serde_json::from_str(&json).expect("decode");
            assert_eq!(decoded, frame);
        }
        let admit_json = serde_json::to_string(&RealtimeClientFrame::Admit {
            admit: Box::new(admit),
        })
        .expect("s");
        assert!(admit_json.contains("\"type\":\"admit\""));
    }

    #[test]
    fn server_frames_round_trip_serde() {
        let event = InvocationEvent {
            schema_version: PROTOCOL_V1,
            request_id: RequestId::new("req-1").expect("request id"),
            attempt_id: AttemptId::new("att-1").expect("attempt id"),
            sequence: 1,
            event: InvocationEventKind::TextDelta {
                text: "partial".into(),
            },
        };
        let frames = [
            RealtimeServerFrame::Admitted {
                session_id: SessionId::new("sess-1").expect("session id"),
                attempt_id: AttemptId::new("att-1").expect("attempt id"),
                worker_id: WorkerId::new("worker-1").expect("worker"),
                node_id: NodeId::new("node-1").expect("node"),
                incarnation_id: IncarnationId::new("inc-1").expect("incarnation"),
                deployment_id: DeploymentId::new("dep-1").expect("deployment"),
                model_generation: ModelGeneration::new(7).expect("generation"),
                output_audio: Some(RealtimeAudioSpec {
                    codec: RealtimeAudioCodec::PcmI16Le,
                    sample_rate: 24_000,
                    channels: 1,
                }),
                bounds: RealtimeSessionBounds {
                    max_frame_bytes: MAX_REALTIME_AUDIO_FRAME_BYTES,
                    max_in_flight_frames: MAX_REALTIME_IN_FLIGHT_FRAMES,
                    max_session_audio_bytes: MAX_REALTIME_SESSION_AUDIO_BYTES,
                },
            },
            RealtimeServerFrame::Pong,
            RealtimeServerFrame::Event { event },
        ];
        for frame in frames {
            let json = serde_json::to_string(&frame).expect("serialize");
            let decoded: RealtimeServerFrame = serde_json::from_str(&json).expect("decode");
            assert_eq!(decoded, frame);
        }
    }

    #[test]
    fn audio_frames_round_trip() {
        let payload: Vec<u8> = (0..64_u16).flat_map(|s| s.to_le_bytes()).collect();
        let frame = encode_realtime_audio_frame(9, true, &payload).expect("encode");
        let (header, decoded) = decode_realtime_audio_frame(&frame).expect("decode");
        assert_eq!(header.sequence, 9);
        assert!(header.is_final);
        assert_eq!(header.payload_len, payload.len());
        assert_eq!(decoded, payload.as_slice());
        assert_eq!(frame.len(), REALTIME_AUDIO_HEADER_BYTES + payload.len());
    }

    #[test]
    fn audio_frames_reject_oversized_and_malformed() {
        let oversized = vec![0_u8; MAX_REALTIME_AUDIO_FRAME_BYTES + 1];
        assert!(matches!(
            encode_realtime_audio_frame(0, false, &oversized),
            Err(RealtimeContractError::AudioFrameTooLarge)
        ));
        assert!(matches!(
            decode_realtime_audio_frame(&[0_u8; 8]),
            Err(RealtimeContractError::MalformedAudioFrame(
                "truncated header"
            ))
        ));
        let mut bad_magic = vec![0_u8; REALTIME_AUDIO_HEADER_BYTES];
        bad_magic[0] = b'X';
        assert!(matches!(
            decode_realtime_audio_frame(&bad_magic),
            Err(RealtimeContractError::MalformedAudioFrame("bad magic"))
        ));
        let mut truncated_payload =
            encode_realtime_audio_frame(0, false, &[1, 2, 3, 4]).expect("encode");
        truncated_payload.pop();
        assert!(matches!(
            decode_realtime_audio_frame(&truncated_payload),
            Err(RealtimeContractError::MalformedAudioFrame(
                "payload length does not match frame length"
            ))
        ));
    }

    #[test]
    fn bounds_reject_out_of_cap_values() {
        let over = RealtimeSessionBounds {
            max_frame_bytes: MAX_REALTIME_AUDIO_FRAME_BYTES + 1,
            max_in_flight_frames: MAX_REALTIME_IN_FLIGHT_FRAMES,
            max_session_audio_bytes: MAX_REALTIME_SESSION_AUDIO_BYTES,
        };
        assert!(matches!(
            over.validate(),
            Err(RealtimeContractError::InvalidBounds)
        ));
        let zero = RealtimeSessionBounds {
            max_frame_bytes: 0,
            max_in_flight_frames: 1,
            max_session_audio_bytes: 1,
        };
        assert!(matches!(
            zero.validate(),
            Err(RealtimeContractError::InvalidBounds)
        ));
    }

    #[test]
    fn worker_feature_and_input_format_variants_are_additive() {
        let features: BTreeSet<WorkerFeature> = BTreeSet::from([
            WorkerFeature::Streaming,
            WorkerFeature::Cancellation,
            WorkerFeature::AttemptQuery,
            WorkerFeature::RealtimeSocket,
        ]);
        let json = serde_json::to_string(&features).expect("serialize");
        assert!(json.contains("realtime_socket"));
        let decoded: BTreeSet<WorkerFeature> = serde_json::from_str(&json).expect("decode");
        assert!(decoded.contains(&WorkerFeature::RealtimeSocket));

        let formats: BTreeSet<crate::types::InputFormat> = BTreeSet::from([
            crate::types::InputFormat::PcmAudio,
            crate::types::InputFormat::Text,
        ]);
        let json = serde_json::to_string(&formats).expect("serialize");
        assert!(json.contains("\"text\""));
        let decoded: BTreeSet<crate::types::InputFormat> =
            serde_json::from_str(&json).expect("decode");
        assert!(decoded.contains(&crate::types::InputFormat::Text));
    }

    /// Static proof that the existing chat invocation contract still
    /// validates after the additive enum extensions in this module's crate.
    #[test]
    fn chat_invocation_contract_unaffected() {
        let input = InvocationInput::Chat {
            input: ChatInput {
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: "hi".into(),
                }],
            },
            parameters: ChatParameters::default(),
        };
        assert_eq!(input.task(), TaskKind::Chat);
        let token = ServiceBearerToken::new("token-value-1").expect("token");
        assert!(token.matches_presented("token-value-1"));
        let _ = CancellationBehavior::Cooperative;
        let _ = OutputFormat::Text;
        let _ = AttemptState::Queued;
    }
}

/// Worker-to-gateway WebSocket close codes for realtime sessions. Values use
/// the WebSocket private-use 4xxx range; the terminal-event vocabulary (not
/// the close code) is the authoritative outcome signal, and orderly session
/// ends close with code 1000.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RealtimeSessionCloseCode {
    ProtocolViolation,
    Unauthorized,
    PolicyDenied,
    UnknownDeployment,
    WrongWorkerIncarnation,
    WrongModelGeneration,
    IncompatibleTask,
    DuplicateAttempt,
    CapacityExhausted,
    ModelNotReady,
    Draining,
    AdmitTimeout,
    Internal,
}

impl RealtimeSessionCloseCode {
    pub const fn code(self) -> u16 {
        match self {
            Self::ProtocolViolation => 4400,
            Self::Unauthorized => 4401,
            Self::PolicyDenied => 4403,
            Self::UnknownDeployment => 4404,
            Self::DuplicateAttempt => 4409,
            Self::WrongModelGeneration => 4410,
            Self::WrongWorkerIncarnation => 4412,
            Self::IncompatibleTask => 4413,
            Self::CapacityExhausted => 4429,
            Self::ModelNotReady => 4453,
            Self::Draining => 4450,
            Self::AdmitTimeout => 4440,
            Self::Internal => 4500,
        }
    }

    pub const fn reason(self) -> &'static str {
        match self {
            Self::ProtocolViolation => "protocol violation",
            Self::Unauthorized => "unauthorized",
            Self::PolicyDenied => "caller policy does not permit realtime sessions",
            Self::UnknownDeployment => "deployment is not loaded",
            Self::DuplicateAttempt => "attempt identity was reused",
            Self::WrongModelGeneration => "model generation changed",
            Self::WrongWorkerIncarnation => "worker incarnation changed",
            Self::IncompatibleTask => "task is incompatible with deployment",
            Self::CapacityExhausted => "worker capacity is exhausted",
            Self::ModelNotReady => "deployment is not ready",
            Self::Draining => "worker is draining",
            Self::AdmitTimeout => "session admission timed out",
            Self::Internal => "internal worker error",
        }
    }
}
