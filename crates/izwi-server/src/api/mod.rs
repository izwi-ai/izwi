//! API routes and handlers

pub mod admin;
pub mod agent;
pub mod chat;
pub mod diarization;
pub mod docs;
pub mod internal;
pub mod media;
pub mod onboarding;
pub mod openai;
pub mod openapi;
pub mod pagination;
pub mod preferences;
pub mod request_context;
mod router;
pub mod saved_voices;
pub mod speech_history;
pub(crate) mod speech_text_upload;
pub mod studio;
pub(crate) mod summary;
pub mod transcription;
pub(crate) mod tts_long_form;
pub mod voice;
pub mod voice_realtime;

pub use router::create_router;
