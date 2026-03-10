//! First-party persisted speech generation resource routes for Text-to-Speech,
//! Voice Design, and Voice Cloning.

mod handlers;

use axum::{extract::DefaultBodyLimit, routing::get, Router};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    const AUDIO_UPLOAD_LIMIT_BYTES: usize = 64 * 1024 * 1024;
    const CANONICAL_TTS_COLLECTION: &str = "/text-to-speech-generations";
    const LEGACY_TTS_COLLECTION: &str = "/text-to-speech/records";
    const CANONICAL_TTS_MEMBER: &str = "/text-to-speech-generations/:record_id";
    const LEGACY_TTS_MEMBER: &str = "/text-to-speech/records/:record_id";
    const CANONICAL_TTS_AUDIO: &str = "/text-to-speech-generations/:record_id/audio";
    const LEGACY_TTS_AUDIO: &str = "/text-to-speech/records/:record_id/audio";
    const CANONICAL_VOICE_DESIGN_COLLECTION: &str = "/voice-design-generations";
    const LEGACY_VOICE_DESIGN_COLLECTION: &str = "/voice-design/records";
    const CANONICAL_VOICE_DESIGN_MEMBER: &str = "/voice-design-generations/:record_id";
    const LEGACY_VOICE_DESIGN_MEMBER: &str = "/voice-design/records/:record_id";
    const CANONICAL_VOICE_DESIGN_AUDIO: &str = "/voice-design-generations/:record_id/audio";
    const LEGACY_VOICE_DESIGN_AUDIO: &str = "/voice-design/records/:record_id/audio";
    const CANONICAL_VOICE_CLONE_COLLECTION: &str = "/voice-clone-generations";
    const LEGACY_VOICE_CLONE_COLLECTION: &str = "/voice-cloning/records";
    const CANONICAL_VOICE_CLONE_MEMBER: &str = "/voice-clone-generations/:record_id";
    const LEGACY_VOICE_CLONE_MEMBER: &str = "/voice-cloning/records/:record_id";
    const CANONICAL_VOICE_CLONE_AUDIO: &str = "/voice-clone-generations/:record_id/audio";
    const LEGACY_VOICE_CLONE_AUDIO: &str = "/voice-cloning/records/:record_id/audio";

    Router::new()
        .route(
            CANONICAL_TTS_COLLECTION,
            get(handlers::list_text_to_speech_records)
                .post(handlers::create_text_to_speech_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            LEGACY_TTS_COLLECTION,
            get(handlers::list_text_to_speech_records)
                .post(handlers::create_text_to_speech_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            CANONICAL_TTS_MEMBER,
            get(handlers::get_text_to_speech_record).delete(handlers::delete_text_to_speech_record),
        )
        .route(
            LEGACY_TTS_MEMBER,
            get(handlers::get_text_to_speech_record).delete(handlers::delete_text_to_speech_record),
        )
        .route(
            CANONICAL_TTS_AUDIO,
            get(handlers::get_text_to_speech_record_audio),
        )
        .route(
            LEGACY_TTS_AUDIO,
            get(handlers::get_text_to_speech_record_audio),
        )
        .route(
            CANONICAL_VOICE_DESIGN_COLLECTION,
            get(handlers::list_voice_design_records)
                .post(handlers::create_voice_design_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            LEGACY_VOICE_DESIGN_COLLECTION,
            get(handlers::list_voice_design_records)
                .post(handlers::create_voice_design_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            CANONICAL_VOICE_DESIGN_MEMBER,
            get(handlers::get_voice_design_record).delete(handlers::delete_voice_design_record),
        )
        .route(
            LEGACY_VOICE_DESIGN_MEMBER,
            get(handlers::get_voice_design_record).delete(handlers::delete_voice_design_record),
        )
        .route(
            CANONICAL_VOICE_DESIGN_AUDIO,
            get(handlers::get_voice_design_record_audio),
        )
        .route(
            LEGACY_VOICE_DESIGN_AUDIO,
            get(handlers::get_voice_design_record_audio),
        )
        .route(
            CANONICAL_VOICE_CLONE_COLLECTION,
            get(handlers::list_voice_cloning_records)
                .post(handlers::create_voice_cloning_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            LEGACY_VOICE_CLONE_COLLECTION,
            get(handlers::list_voice_cloning_records)
                .post(handlers::create_voice_cloning_record)
                .layer(DefaultBodyLimit::max(AUDIO_UPLOAD_LIMIT_BYTES)),
        )
        .route(
            CANONICAL_VOICE_CLONE_MEMBER,
            get(handlers::get_voice_cloning_record).delete(handlers::delete_voice_cloning_record),
        )
        .route(
            LEGACY_VOICE_CLONE_MEMBER,
            get(handlers::get_voice_cloning_record).delete(handlers::delete_voice_cloning_record),
        )
        .route(
            CANONICAL_VOICE_CLONE_AUDIO,
            get(handlers::get_voice_cloning_record_audio),
        )
        .route(
            LEGACY_VOICE_CLONE_AUDIO,
            get(handlers::get_voice_cloning_record_audio),
        )
}
