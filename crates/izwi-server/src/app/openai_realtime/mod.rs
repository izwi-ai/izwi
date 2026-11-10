//! OpenAI Realtime (GA) over WebSocket at `/v1/realtime`: a stock OpenAI
//! Realtime client (openai-python, openai-node, LiveKit Agents, Pipecat)
//! holds a speech-to-speech session with Izwi's local models.
//!
//! This is a front-end over the voice session core in `voice_realtime`
//! (VAD and endpointing, the reply pipeline, the spoken ledger, per-stage
//! admission), not a second voice engine. `?model=` picks a realtime profile:
//! a modular ASR → chat → TTS stack or a unified LFM-Audio model.

pub(crate) mod audio;
pub(crate) mod connection;
pub(crate) mod events;
pub(crate) mod items;
pub(crate) mod live;
pub(crate) mod persist;
pub(crate) mod profiles;
pub(crate) mod respond;
pub(crate) mod secrets;
pub(crate) mod session_config;
pub(crate) mod transcribe;
pub(crate) mod voices;

pub(crate) use connection::{handle_socket, SessionKind};

#[cfg(test)]
mod socket_tests;
