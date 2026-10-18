//! The realtime wire formats the gateway relays without running a model:
//! the client PCM16 frame layout, and the transcription endpoint's
//! protocol negotiation, control events and binary frames. izwi-server's
//! realtime endpoints use the same definitions.

use serde::Deserialize;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::app::realtime_protocol::TRANSCRIPTION_REALTIME_VERSION;

/// Bytes in the client PCM16 frame header: magic, version, kind, two
/// reserved bytes, sample rate and frame sequence.
pub const WS_BIN_CLIENT_HEADER_LEN: usize = 16;
pub const WS_BIN_VERSION: u8 = 1;
pub const WS_BIN_KIND_CLIENT_PCM16: u8 = 1;

pub const LEGACY_REALTIME_PROTOCOL: &str = "transcription_realtime_v2";
pub const TYPED_REALTIME_PROTOCOL: &str = "transcription_realtime";

pub const WS_BIN_MAGIC: &[u8; 4] = b"ITRW";

/// One client PCM16 audio frame.
#[derive(Debug)]
pub struct ClientPcm16Frame {
    pub frame_seq: u32,
    pub sample_rate: u32,
    pub payload: Vec<u8>,
}

/// Parse a client PCM16 frame carrying `magic`; `endpoint` names the
/// endpoint in the error for a frame without the header.
pub fn parse_client_pcm16_frame(
    data: &[u8],
    magic: &[u8; 4],
    endpoint: &str,
) -> Result<ClientPcm16Frame, String> {
    if data.len() < WS_BIN_CLIENT_HEADER_LEN || &data[..4] != magic {
        return Err(format!(
            "Unexpected binary message (missing {endpoint} frame header)"
        ));
    }
    let version = data[4];
    if version != WS_BIN_VERSION {
        return Err(format!("Unsupported binary frame version {version}"));
    }
    match data[5] {
        WS_BIN_KIND_CLIENT_PCM16 => Ok(ClientPcm16Frame {
            sample_rate: u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            frame_seq: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            payload: data[WS_BIN_CLIENT_HEADER_LEN..].to_vec(),
        }),
        other => Err(format!("Unsupported binary frame kind {other}")),
    }
}

pub fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientEvent {
    SessionStart {
        #[serde(default)]
        model_id: Option<String>,
        #[serde(default)]
        language: Option<String>,
        #[serde(default)]
        protocol: Option<String>,
        #[serde(default)]
        version: Option<u16>,
        #[serde(default)]
        resume_from_event_id: Option<u64>,
    },
    SessionStop,
    Ping {
        #[serde(default)]
        timestamp_ms: Option<u64>,
    },
}

#[derive(Debug)]
pub enum BinaryMessageKind {
    ClientPcm16Frame {
        frame_seq: u32,
        sample_rate: u32,
        payload: Vec<u8>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionWireProtocol {
    LegacyV2,
    TypedV3,
}

pub fn negotiate_transcription_protocol(
    protocol: Option<&str>,
    version: Option<u16>,
    resume_from_event_id: Option<u64>,
) -> Result<TranscriptionWireProtocol, String> {
    match (protocol.map(str::trim), version) {
        (None, None) => Ok(TranscriptionWireProtocol::LegacyV2),
        (Some(TYPED_REALTIME_PROTOCOL), Some(TRANSCRIPTION_REALTIME_VERSION)) => {
            if resume_from_event_id.is_some() {
                return Err(
                    "transcription_realtime v3 sessions are process-owned and non-resumable"
                        .to_string(),
                );
            }
            Ok(TranscriptionWireProtocol::TypedV3)
        }
        (Some(protocol), Some(version)) => Err(format!(
            "Unsupported realtime protocol negotiation `{protocol}` version {version}"
        )),
        _ => {
            Err("Realtime protocol negotiation requires both `protocol` and `version`".to_string())
        }
    }
}

pub fn parse_binary_message(data: &[u8]) -> Result<BinaryMessageKind, String> {
    let ClientPcm16Frame {
        frame_seq,
        sample_rate,
        payload,
    } = parse_client_pcm16_frame(data, WS_BIN_MAGIC, "transcription")?;
    Ok(BinaryMessageKind::ClientPcm16Frame {
        frame_seq,
        sample_rate,
        payload,
    })
}
