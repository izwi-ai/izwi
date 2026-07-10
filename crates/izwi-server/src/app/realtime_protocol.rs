//! Typed wire contracts for realtime transcription and voice sessions.
//!
//! The current websocket handlers still speak their legacy preview protocols.
//! These types define the versioned contract that later adapters can use without
//! coupling protocol semantics to socket orchestration or inference internals.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const TRANSCRIPTION_REALTIME_VERSION: u16 = 3;
pub const VOICE_REALTIME_VERSION: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeProtocol {
    TranscriptionRealtime,
    VoiceRealtime,
}

impl RealtimeProtocol {
    pub const fn current_version(self) -> u16 {
        match self {
            Self::TranscriptionRealtime => TRANSCRIPTION_REALTIME_VERSION,
            Self::VoiceRealtime => VOICE_REALTIME_VERSION,
        }
    }
}

/// Metadata shared by every JSON client and server event.
///
/// `event_id` is monotonically increasing for the logical session, including
/// across connection epochs. `sequence` starts at zero for each connection
/// epoch and must increase by exactly one for each event in that epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeEventEnvelope<E> {
    pub protocol: RealtimeProtocol,
    pub version: u16,
    pub event_id: u64,
    pub sequence: u64,
    pub session_id: String,
    pub connection_epoch: u64,
    pub timestamp_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub utterance_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
    #[serde(flatten)]
    pub event: E,
}

impl<E> RealtimeEventEnvelope<E> {
    pub fn position(&self) -> RealtimeEventPosition {
        RealtimeEventPosition {
            event_id: self.event_id,
            sequence: self.sequence,
            connection_epoch: self.connection_epoch,
        }
    }

    pub fn validate_protocol_version(&self) -> Result<(), RealtimeEnvelopeError> {
        let expected = self.protocol.current_version();
        if self.version != expected {
            return Err(RealtimeEnvelopeError::UnsupportedVersion {
                protocol: self.protocol,
                expected,
                actual: self.version,
            });
        }
        if self.session_id.trim().is_empty() {
            return Err(RealtimeEnvelopeError::MissingSessionId);
        }
        Ok(())
    }

    pub fn validate_successor<P>(
        &self,
        previous: &RealtimeEventEnvelope<P>,
    ) -> Result<(), RealtimeEnvelopeError> {
        if self.protocol != previous.protocol {
            return Err(RealtimeEnvelopeError::ProtocolChanged);
        }
        if self.session_id != previous.session_id {
            return Err(RealtimeEnvelopeError::SessionChanged);
        }
        self.validate_protocol_version()?;
        previous.validate_protocol_version()?;
        if self.event_id <= previous.event_id {
            return Err(RealtimeEnvelopeError::EventIdNotIncreasing {
                previous: previous.event_id,
                actual: self.event_id,
            });
        }

        match self.connection_epoch.cmp(&previous.connection_epoch) {
            std::cmp::Ordering::Less => Err(RealtimeEnvelopeError::EpochRegressed {
                previous: previous.connection_epoch,
                actual: self.connection_epoch,
            }),
            std::cmp::Ordering::Equal => {
                let expected = previous.sequence.saturating_add(1);
                if self.sequence != expected {
                    return Err(RealtimeEnvelopeError::InvalidSequence {
                        expected,
                        actual: self.sequence,
                    });
                }
                Ok(())
            }
            std::cmp::Ordering::Greater => {
                let expected_epoch = previous.connection_epoch.saturating_add(1);
                if self.connection_epoch != expected_epoch {
                    return Err(RealtimeEnvelopeError::EpochSkipped {
                        expected: expected_epoch,
                        actual: self.connection_epoch,
                    });
                }
                if self.sequence != 0 {
                    return Err(RealtimeEnvelopeError::EpochMustStartAtZero {
                        actual: self.sequence,
                    });
                }
                Ok(())
            }
        }
    }
}

pub type RealtimeClientEnvelope = RealtimeEventEnvelope<RealtimeClientEvent>;
pub type RealtimeServerEnvelope = RealtimeEventEnvelope<RealtimeServerEvent>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeEventPosition {
    pub event_id: u64,
    pub sequence: u64,
    pub connection_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RealtimeEnvelopeError {
    #[error("realtime session id is required")]
    MissingSessionId,
    #[error("realtime protocol changed within a session")]
    ProtocolChanged,
    #[error("realtime session id changed within a session")]
    SessionChanged,
    #[error("unsupported {protocol:?} version: expected {expected}, got {actual}")]
    UnsupportedVersion {
        protocol: RealtimeProtocol,
        expected: u16,
        actual: u16,
    },
    #[error("event id must increase: previous {previous}, got {actual}")]
    EventIdNotIncreasing { previous: u64, actual: u64 },
    #[error("connection epoch regressed: previous {previous}, got {actual}")]
    EpochRegressed { previous: u64, actual: u64 },
    #[error("connection epoch skipped: expected {expected}, got {actual}")]
    EpochSkipped { expected: u64, actual: u64 },
    #[error("event sequence must be {expected}, got {actual}")]
    InvalidSequence { expected: u64, actual: u64 },
    #[error("a new connection epoch must start at sequence zero, got {actual}")]
    EpochMustStartAtZero { actual: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RealtimeClientEvent {
    SessionStart {
        requested_version: u16,
        #[serde(skip_serializing_if = "Option::is_none")]
        resume_from_event_id: Option<u64>,
    },
    AudioFrame {
        frame_sequence: u64,
        sample_rate: u32,
        sample_count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        capture_timestamp_ms: Option<u64>,
    },
    AudioStop,
    Interrupt {
        reason: RealtimeInterruptionReason,
    },
    Ping {
        #[serde(skip_serializing_if = "Option::is_none")]
        client_timestamp_ms: Option<u64>,
    },
    Close {
        reason: RealtimeCloseReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RealtimeServerEvent {
    SessionReady {
        accepted_version: u16,
        owner_instance_id: String,
        resumable: bool,
        resume_window_ms: u64,
    },
    SessionStarted,
    AudioAccepted {
        frame_sequence: u64,
        buffer_depth_samples: usize,
        ingress_queue_depth: usize,
    },
    AudioGap {
        expected_frame_sequence: u64,
        received_frame_sequence: u64,
        missing_frames: u64,
        action: RealtimeAudioGapAction,
    },
    SpeechStarted,
    SpeechEnded {
        reason: RealtimeSpeechEndReason,
    },
    TranscriptPartial {
        /// Complete replaceable hypothesis, never an append-only delta.
        text: String,
        revision: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    TranscriptStable {
        /// Complete transcript with an append-only committed prefix.
        text: String,
        revision: u64,
        stable_prefix_chars: usize,
    },
    TranscriptCorrection {
        /// Complete authoritative replacement for `replaces_revision`.
        text: String,
        revision: u64,
        replaces_revision: u64,
        reason: String,
    },
    TranscriptFinal {
        /// Exactly one authoritative terminal transcript per segment.
        text: String,
        revision: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    AssistantTextStarted,
    AssistantTextPartial {
        /// Complete replaceable assistant-text hypothesis, never an append-only delta.
        text: String,
        revision: u64,
    },
    AssistantTextFinal {
        text: String,
    },
    AssistantAudioStarted {
        sample_rate: u32,
        channels: u16,
        format: RealtimeAudioFormat,
    },
    AssistantAudioChunk {
        chunk_sequence: u64,
        sample_count: usize,
        is_final: bool,
    },
    AssistantAudioCompleted {
        last_chunk_sequence: u64,
    },
    Interruption {
        reason: RealtimeInterruptionReason,
        cutoff_event_id: u64,
        cutoff_sequence: u64,
    },
    TurnCompleted {
        status: RealtimeTurnStatus,
    },
    RecoverableError {
        code: RealtimeErrorCode,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        retry_after_ms: Option<u64>,
    },
    FatalError {
        code: RealtimeErrorCode,
        message: String,
        close: RealtimeClose,
    },
    Closing {
        close: RealtimeClose,
    },
    Closed {
        close: RealtimeClose,
    },
    Pong {
        #[serde(skip_serializing_if = "Option::is_none")]
        client_timestamp_ms: Option<u64>,
        server_timestamp_ms: u64,
    },
}

impl RealtimeServerEvent {
    pub const fn finality(&self) -> RealtimeEventFinality {
        match self {
            Self::TranscriptPartial { .. }
            | Self::AssistantTextStarted
            | Self::AssistantTextPartial { .. }
            | Self::AssistantAudioStarted { .. }
            | Self::AssistantAudioChunk { .. } => RealtimeEventFinality::Intermediate,
            Self::TranscriptStable { .. } => RealtimeEventFinality::Stable,
            Self::TranscriptCorrection { .. } => RealtimeEventFinality::Correction,
            Self::TranscriptFinal { .. }
            | Self::AssistantTextFinal { .. }
            | Self::AssistantAudioCompleted { .. } => RealtimeEventFinality::SegmentFinal,
            Self::Interruption { .. } => RealtimeEventFinality::TurnCutoff,
            Self::TurnCompleted { .. } => RealtimeEventFinality::TurnFinal,
            Self::Closed { .. } => RealtimeEventFinality::SessionFinal,
            _ => RealtimeEventFinality::NotApplicable,
        }
    }

    pub const fn delivery_class(&self) -> RealtimeDeliveryClass {
        match self {
            Self::TranscriptPartial { .. }
            | Self::TranscriptStable { .. }
            | Self::AssistantTextPartial { .. } => RealtimeDeliveryClass::CoalescibleSnapshot,
            Self::AssistantAudioChunk { .. } => RealtimeDeliveryClass::LosslessStream,
            Self::Pong { .. } | Self::AudioAccepted { .. } => RealtimeDeliveryClass::Droppable,
            _ => RealtimeDeliveryClass::Critical,
        }
    }

    pub const fn requires_connection_close(&self) -> bool {
        matches!(
            self,
            Self::FatalError { .. } | Self::Closing { .. } | Self::Closed { .. }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeEventFinality {
    NotApplicable,
    Intermediate,
    Stable,
    Correction,
    SegmentFinal,
    TurnCutoff,
    TurnFinal,
    SessionFinal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeDeliveryClass {
    /// Must be delivered or terminate the connection with an explicit error.
    Critical,
    /// May replace an older unsent snapshot for the same logical stream.
    CoalescibleSnapshot,
    /// Must preserve order and content; saturation interrupts the stream.
    LosslessStream,
    /// May be omitted when diagnostic traffic would delay user-visible events.
    Droppable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeAudioGapAction {
    Continue,
    ResetSegment,
    CloseSession,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeSpeechEndReason {
    Silence,
    MaxDuration,
    ClientPause,
    StreamStopped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeInterruptionReason {
    ClientRequest,
    BargeIn,
    PreemptedByNewTurn,
    Backpressure,
    SessionClosing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeTurnStatus {
    Ok,
    NoInput,
    Interrupted,
    Error,
    Timeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeAudioFormat {
    PcmI16,
    PcmF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeErrorCode {
    InvalidMessage,
    InvalidTransition,
    UnsupportedVersion,
    AudioGap,
    BufferLimit,
    Backpressure,
    ModelUnavailable,
    InferenceFailed,
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealtimeClose {
    pub code: RealtimeCloseCode,
    pub reason: RealtimeCloseReason,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeCloseCode {
    Normal,
    ProtocolError,
    PolicyViolation,
    InternalError,
    ServiceRestart,
    TryAgainLater,
}

impl RealtimeCloseCode {
    pub const fn websocket_code(self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::ProtocolError => 1002,
            Self::PolicyViolation => 1008,
            Self::InternalError => 1011,
            Self::ServiceRestart => 1012,
            Self::TryAgainLater => 1013,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RealtimeCloseReason {
    Normal,
    ClientRequest,
    IdleTimeout,
    UnsupportedVersion,
    ProtocolError,
    Overloaded,
    ServerShutdown,
    InternalError,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn server_envelope(
        event_id: u64,
        sequence: u64,
        event: RealtimeServerEvent,
    ) -> RealtimeServerEnvelope {
        RealtimeEventEnvelope {
            protocol: RealtimeProtocol::TranscriptionRealtime,
            version: TRANSCRIPTION_REALTIME_VERSION,
            event_id,
            sequence,
            session_id: "session-1".to_string(),
            connection_epoch: 0,
            timestamp_ms: 1_725_000_000_123,
            utterance_id: Some("utterance-1".to_string()),
            turn_id: None,
            segment_id: Some("segment-1".to_string()),
            event,
        }
    }

    #[test]
    fn protocol_versions_are_explicit() {
        assert_eq!(RealtimeProtocol::TranscriptionRealtime.current_version(), 3);
        assert_eq!(RealtimeProtocol::VoiceRealtime.current_version(), 2);
    }

    #[test]
    fn transcript_stable_wire_shape_is_golden() {
        let envelope = server_envelope(
            9,
            8,
            RealtimeServerEvent::TranscriptStable {
                text: "hello world".to_string(),
                revision: 4,
                stable_prefix_chars: 5,
            },
        );

        let serialized = serde_json::to_string(&envelope).expect("serialize stable event");
        assert_eq!(
            serialized,
            r#"{"protocol":"transcription_realtime","version":3,"event_id":9,"sequence":8,"session_id":"session-1","connection_epoch":0,"timestamp_ms":1725000000123,"utterance_id":"utterance-1","segment_id":"segment-1","type":"transcript_stable","data":{"text":"hello world","revision":4,"stable_prefix_chars":5}}"#
        );
    }

    #[test]
    fn client_session_start_wire_shape_is_golden() {
        let envelope = RealtimeClientEnvelope {
            protocol: RealtimeProtocol::VoiceRealtime,
            version: VOICE_REALTIME_VERSION,
            event_id: 1,
            sequence: 0,
            session_id: "client-session".to_string(),
            connection_epoch: 0,
            timestamp_ms: 42,
            utterance_id: None,
            turn_id: None,
            segment_id: None,
            event: RealtimeClientEvent::SessionStart {
                requested_version: VOICE_REALTIME_VERSION,
                resume_from_event_id: None,
            },
        };

        assert_eq!(
            serde_json::to_value(&envelope).expect("serialize client event"),
            json!({
                "protocol": "voice_realtime",
                "version": 2,
                "event_id": 1,
                "sequence": 0,
                "session_id": "client-session",
                "connection_epoch": 0,
                "timestamp_ms": 42,
                "type": "session_start",
                "data": { "requested_version": 2 }
            })
        );
    }

    #[test]
    fn optional_identifiers_are_omitted_instead_of_serialized_as_null() {
        let mut envelope = server_envelope(1, 0, RealtimeServerEvent::SessionStarted);
        envelope.utterance_id = None;
        envelope.segment_id = None;
        let value = serde_json::to_value(envelope).expect("serialize event");

        assert!(value.get("utterance_id").is_none());
        assert!(value.get("turn_id").is_none());
        assert!(value.get("segment_id").is_none());
    }

    #[test]
    fn event_order_is_strict_within_an_epoch() {
        let first = server_envelope(10, 4, RealtimeServerEvent::SessionStarted);
        let next = server_envelope(11, 5, RealtimeServerEvent::SpeechStarted);
        assert_eq!(next.validate_successor(&first), Ok(()));

        let duplicate_sequence = server_envelope(12, 5, RealtimeServerEvent::SpeechStarted);
        assert_eq!(
            duplicate_sequence.validate_successor(&next),
            Err(RealtimeEnvelopeError::InvalidSequence {
                expected: 6,
                actual: 5,
            })
        );
    }

    #[test]
    fn reconnect_epoch_resets_sequence_but_not_event_id() {
        let first = server_envelope(10, 4, RealtimeServerEvent::SessionStarted);
        let mut reconnected = server_envelope(
            11,
            0,
            RealtimeServerEvent::SessionReady {
                accepted_version: TRANSCRIPTION_REALTIME_VERSION,
                owner_instance_id: "server-1".to_string(),
                resumable: false,
                resume_window_ms: 0,
            },
        );
        reconnected.connection_epoch = 1;

        assert_eq!(reconnected.validate_successor(&first), Ok(()));

        reconnected.sequence = 1;
        assert_eq!(
            reconnected.validate_successor(&first),
            Err(RealtimeEnvelopeError::EpochMustStartAtZero { actual: 1 })
        );
    }

    #[test]
    fn protocol_and_session_cannot_change_midstream() {
        let first = server_envelope(1, 0, RealtimeServerEvent::SessionStarted);
        let mut next = server_envelope(2, 1, RealtimeServerEvent::SpeechStarted);
        next.session_id = "other-session".to_string();
        assert_eq!(
            next.validate_successor(&first),
            Err(RealtimeEnvelopeError::SessionChanged)
        );

        next.session_id = first.session_id.clone();
        next.protocol = RealtimeProtocol::VoiceRealtime;
        assert_eq!(
            next.validate_successor(&first),
            Err(RealtimeEnvelopeError::ProtocolChanged)
        );
    }

    #[test]
    fn event_semantics_distinguish_finality_and_delivery() {
        let partial = RealtimeServerEvent::TranscriptPartial {
            text: "hel".to_string(),
            revision: 1,
            language: None,
        };
        let correction = RealtimeServerEvent::TranscriptCorrection {
            text: "hello".to_string(),
            revision: 3,
            replaces_revision: 2,
            reason: "decoder_revision".to_string(),
        };
        let audio = RealtimeServerEvent::AssistantAudioChunk {
            chunk_sequence: 7,
            sample_count: 4_800,
            is_final: false,
        };
        let closed = RealtimeServerEvent::Closed {
            close: normal_close(),
        };

        assert_eq!(partial.finality(), RealtimeEventFinality::Intermediate);
        assert_eq!(
            partial.delivery_class(),
            RealtimeDeliveryClass::CoalescibleSnapshot
        );
        assert_eq!(correction.finality(), RealtimeEventFinality::Correction);
        assert_eq!(correction.delivery_class(), RealtimeDeliveryClass::Critical);
        assert_eq!(
            audio.delivery_class(),
            RealtimeDeliveryClass::LosslessStream
        );
        assert_eq!(closed.finality(), RealtimeEventFinality::SessionFinal);
        assert!(closed.requires_connection_close());
    }

    #[test]
    fn recoverable_and_fatal_errors_have_distinct_close_semantics() {
        let recoverable = RealtimeServerEvent::RecoverableError {
            code: RealtimeErrorCode::AudioGap,
            message: "one input frame was missing".to_string(),
            retry_after_ms: None,
        };
        let fatal = RealtimeServerEvent::FatalError {
            code: RealtimeErrorCode::Backpressure,
            message: "client remained slow beyond the deadline".to_string(),
            close: RealtimeClose {
                code: RealtimeCloseCode::TryAgainLater,
                reason: RealtimeCloseReason::Overloaded,
                message: "outbound stream saturated".to_string(),
                retryable: true,
            },
        };

        assert!(!recoverable.requires_connection_close());
        assert!(fatal.requires_connection_close());
        assert_eq!(RealtimeCloseCode::TryAgainLater.websocket_code(), 1013);
        assert_eq!(RealtimeCloseCode::ProtocolError.websocket_code(), 1002);
    }

    #[test]
    fn all_server_event_families_round_trip() {
        let events = vec![
            RealtimeServerEvent::AudioAccepted {
                frame_sequence: 2,
                buffer_depth_samples: 320,
                ingress_queue_depth: 1,
            },
            RealtimeServerEvent::AudioGap {
                expected_frame_sequence: 3,
                received_frame_sequence: 5,
                missing_frames: 2,
                action: RealtimeAudioGapAction::ResetSegment,
            },
            RealtimeServerEvent::SpeechEnded {
                reason: RealtimeSpeechEndReason::ClientPause,
            },
            RealtimeServerEvent::TranscriptFinal {
                text: "done".to_string(),
                revision: 8,
                language: Some("en".to_string()),
            },
            RealtimeServerEvent::AssistantTextFinal {
                text: "answer".to_string(),
            },
            RealtimeServerEvent::AssistantAudioCompleted {
                last_chunk_sequence: 12,
            },
            RealtimeServerEvent::Interruption {
                reason: RealtimeInterruptionReason::BargeIn,
                cutoff_event_id: 44,
                cutoff_sequence: 43,
            },
            RealtimeServerEvent::TurnCompleted {
                status: RealtimeTurnStatus::Interrupted,
            },
            RealtimeServerEvent::Closing {
                close: normal_close(),
            },
        ];

        for (index, event) in events.into_iter().enumerate() {
            let envelope = server_envelope(index as u64 + 1, index as u64, event);
            let encoded = serde_json::to_vec(&envelope).expect("serialize server envelope");
            let decoded: RealtimeServerEnvelope =
                serde_json::from_slice(&encoded).expect("deserialize server envelope");
            assert_eq!(decoded, envelope);
        }
    }

    fn normal_close() -> RealtimeClose {
        RealtimeClose {
            code: RealtimeCloseCode::Normal,
            reason: RealtimeCloseReason::Normal,
            message: "session complete".to_string(),
            retryable: false,
        }
    }
}
