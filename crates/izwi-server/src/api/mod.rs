//! API routes and handlers

pub mod admin;
pub mod agent;
pub(crate) mod asr_task;
pub(crate) mod audio_payload;
pub mod chat;
pub mod diarization;
pub use izwi_gateway::api::docs;
pub(crate) mod idempotency;
pub mod internal;
pub mod jobs;
pub mod media;
pub mod onboarding;
pub mod openai;
pub mod openapi;
pub mod pagination;
pub mod preferences;
pub mod request_context;
mod router;
pub mod saved_voices;
pub(crate) mod sentence_split;
pub mod speech_history;
pub(crate) mod speech_text_upload;
pub mod studio;
pub(crate) mod summary;
pub(crate) mod transcript_alignment;
pub mod transcription;
pub(crate) mod tts_long_form;
pub(crate) mod tts_policy;
pub mod voice;
pub mod voice_realtime;

pub use router::create_router;
pub(crate) use router::DESKTOP_CORS_ORIGINS;
