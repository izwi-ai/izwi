//! Custom voices and voice consents in OpenAI's shape (`/v1/audio/voices`,
//! `/v1/audio/voice_consents`), also serving SGLang-Omni and vLLM-Omni voice
//! clients, over Izwi's saved-voice store (`/v1/voices`).
//!
//! A voice made here is a saved voice: `/v1/audio/speech` speaks it as
//! `voice: {"id": ...}` (or by name, for a model that clones and has no
//! built-in voices), and `/v1/realtime` as `session.audio.output.voice`.
//!
//! What creating a voice needs depends on `IZWI_VOICE_CONSENT_MODE`:
//! - `attestation`: any non-empty `consent` (OpenAI's consent id, or
//!   SGLang-Omni's free text), kept as the voice's provenance;
//! - `recording`: `consent` names a consent recording made here;
//! - `verified`: that recording's transcript matched a published consent
//!   phrase (`GET /v1/audio/consent_phrases`).
//!
//! Unset, a server protected by an API key uses `recording` and an open
//! local server `attestation`. Deleting a consent revokes every voice made
//! with it.

use axum::{
    extract::{Extension, Multipart, Path, Query, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::api::audio_payload::{
    inspect_audio_payload_with_diagnostics, read_multipart_audio_file_payload, AudioPayload,
};
use crate::api::request_context::RequestContext;
use crate::env_config::VoiceConsentMode;
use crate::error::{ApiError, ApiJson};
use crate::saved_voice_store::{NewSavedVoice, NewVoiceConsent, SavedVoice, VoiceConsent};
use crate::state::AppState;

/// OpenAI's upload cap for a consent recording or an audio sample.
pub(crate) const VOICE_UPLOAD_LIMIT_BYTES: usize = 10 * 1024 * 1024;

/// OpenAI's built-in voice names (and `default`): a voice may not take one,
/// so `voice: "alloy"` always means the model's own voice.
const RESERVED_VOICE_NAMES: &[&str] = &[
    "default", "alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage",
    "shimmer", "verse", "marin", "cedar",
];

/// One phrase a consent recording says, per language.
#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct ConsentPhrase {
    pub(crate) language: &'static str,
    pub(crate) text: &'static str,
}

/// The phrases a `verified` consent recording must say. Izwi publishes its
/// own (OpenAI's list is not in its API); a language joins with a reviewed
/// phrase.
pub(crate) const CONSENT_PHRASES: &[ConsentPhrase] = &[ConsentPhrase {
    language: "en",
    text: "I am the owner of this voice and I consent to Izwi creating a synthetic version of it.",
}];

/// How much of a consent phrase a transcript must match (word-level).
const CONSENT_MATCH_THRESHOLD: f32 = 0.8;

pub(crate) fn consent_mode() -> VoiceConsentMode {
    let tuning = crate::env_config::tuning();
    tuning
        .speech_api
        .voice_consent_mode
        .unwrap_or(if tuning.public_probe_details {
            VoiceConsentMode::Attestation
        } else {
            VoiceConsentMode::Recording
        })
}

fn unix_secs(millis: u64) -> u64 {
    millis / 1_000
}

fn store_error(err: anyhow::Error) -> ApiError {
    crate::error::store_api_error(err, "Saved voice storage error")
}

// ---------------------------------------------------------------- consents

#[derive(Debug, Serialize)]
pub struct VoiceConsentResource {
    object: &'static str,
    id: String,
    name: String,
    language: String,
    created_at: u64,
    /// izwi: `recorded`, `verified` or `rejected`.
    status: String,
    /// izwi: the transcript check, when one ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    verification: Option<serde_json::Value>,
}

impl From<VoiceConsent> for VoiceConsentResource {
    fn from(consent: VoiceConsent) -> Self {
        Self {
            object: "audio.voice_consent",
            id: consent.id,
            name: consent.name,
            language: consent.language,
            created_at: unix_secs(consent.created_at),
            status: consent.status,
            verification: consent.verification,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct VoiceConsentList {
    object: &'static str,
    data: Vec<VoiceConsentResource>,
    first_id: Option<String>,
    last_id: Option<String>,
    has_more: bool,
}

#[derive(Debug, Serialize)]
pub struct DeletedResource {
    id: String,
    object: &'static str,
    deleted: bool,
    /// SGLang-Omni's field.
    #[serde(skip_serializing_if = "Option::is_none")]
    success: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default)]
    after: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    /// SGLang-Omni: list only uploaded voice names.
    #[serde(default)]
    names_only: Option<bool>,
}

async fn text_field(
    field: axum::extract::multipart::Field<'_>,
    name: &str,
) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map(|text| text.trim().to_string())
        .map_err(|err| {
            ApiError::bad_request(format!("Failed reading multipart '{name}' field: {err}"))
        })
}

/// `POST /v1/audio/voice_consents`: multipart `name`, `recording` and
/// `language` (BCP 47).
pub async fn create_voice_consent(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<VoiceConsentResource>, ApiError> {
    let (mut name, mut language, mut recording) = (None, None, None::<AudioPayload>);
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(format!("Invalid multipart payload: {err}")))?
    {
        match field.name().unwrap_or_default() {
            "name" => name = Some(text_field(field, "name").await?),
            "language" => language = Some(text_field(field, "language").await?),
            "recording" => {
                recording = read_multipart_audio_file_payload(
                    field,
                    "audio.voice_consents",
                    "recording",
                    VOICE_UPLOAD_LIMIT_BYTES,
                )
                .await?;
            }
            _ => {}
        }
    }
    let name = name
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::invalid_param("name", "`name` is required"))?;
    let language = language
        .filter(|language| !language.is_empty())
        .ok_or_else(|| {
            ApiError::invalid_param("language", "`language` is required (BCP 47, e.g. en-US)")
        })?;
    let recording =
        recording.ok_or_else(|| ApiError::invalid_param("recording", "`recording` is required"))?;
    inspect_audio_payload_with_diagnostics("audio.voice_consents", &recording)?;
    let consent = state
        .saved_voice_store
        .create_consent(NewVoiceConsent {
            name,
            language: language.clone(),
            recording_mime_type: recording
                .content_type_hint()
                .unwrap_or("audio/wav")
                .to_string(),
            recording_filename: recording.filename.clone(),
            recording_bytes: recording.bytes.clone(),
        })
        .await
        .map_err(store_error)?;
    let consent = if consent_mode() == VoiceConsentMode::Verified {
        verify_consent(&state, consent, &recording.bytes).await?
    } else {
        consent
    };
    Ok(Json(consent.into()))
}

/// Transcribes the recording and checks it says the language's consent
/// phrase; records `verified` or `rejected` (or leaves `recorded` when no
/// ASR model can run).
async fn verify_consent(
    state: &AppState,
    consent: VoiceConsent,
    recording: &[u8],
) -> Result<VoiceConsent, ApiError> {
    let base_language = consent
        .language
        .split(['-', '_'])
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let Some(phrase) = CONSENT_PHRASES
        .iter()
        .find(|phrase| phrase.language == base_language)
    else {
        let verification = serde_json::json!({
            "error": format!("No consent phrase is published for language `{}`", consent.language),
        });
        return state
            .saved_voice_store
            .set_consent_verification(&consent.id, "rejected", &verification)
            .await
            .map_err(store_error)?
            .ok_or_else(|| ApiError::internal("Consent vanished during verification"));
    };
    let (status, verification) = match state
        .runtime
        .asr_transcribe_bytes(recording, None, Some(base_language.as_str()))
        .await
    {
        Ok(transcript) => {
            let score = phrase_match(&transcript.text, phrase.text);
            let status = if score >= CONSENT_MATCH_THRESHOLD {
                "verified"
            } else {
                "rejected"
            };
            (
                status,
                serde_json::json!({
                    "transcript": transcript.text,
                    "phrase": phrase.text,
                    "score": score,
                    "threshold": CONSENT_MATCH_THRESHOLD,
                }),
            )
        }
        Err(err) => (
            "recorded",
            serde_json::json!({ "error": format!("The recording could not be transcribed: {err}") }),
        ),
    };
    state
        .saved_voice_store
        .set_consent_verification(&consent.id, status, &verification)
        .await
        .map_err(store_error)?
        .ok_or_else(|| ApiError::internal("Consent vanished during verification"))
}

fn words(text: &str) -> Vec<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// 1 minus the word edit distance between `transcript` and `phrase`, over
/// the phrase's length (floored at 0).
pub(crate) fn phrase_match(transcript: &str, phrase: &str) -> f32 {
    let (said, expected) = (words(transcript), words(phrase));
    if expected.is_empty() {
        return 0.0;
    }
    let mut previous: Vec<usize> = (0..=expected.len()).collect();
    for (i, word) in said.iter().enumerate() {
        let mut current = vec![i + 1; expected.len() + 1];
        for (j, target) in expected.iter().enumerate() {
            let substitution = previous[j] + usize::from(word != target);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        previous = current;
    }
    let distance = previous[expected.len()] as f32;
    (1.0 - distance / expected.len() as f32).max(0.0)
}

pub async fn list_voice_consents(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<Json<VoiceConsentList>, ApiError> {
    let (consents, has_more) = state
        .saved_voice_store
        .list_consents(query.after.as_deref(), query.limit.unwrap_or(20))
        .await
        .map_err(store_error)?;
    let data: Vec<VoiceConsentResource> = consents.into_iter().map(Into::into).collect();
    Ok(Json(VoiceConsentList {
        object: "list",
        first_id: data.first().map(|consent| consent.id.clone()),
        last_id: data.last().map(|consent| consent.id.clone()),
        data,
        has_more,
    }))
}

async fn consent_or_404(state: &AppState, id: &str) -> Result<VoiceConsent, ApiError> {
    state
        .saved_voice_store
        .get_consent(id)
        .await
        .map_err(store_error)?
        .ok_or_else(|| ApiError::not_found(format!("No voice consent with id {id}")))
}

pub async fn get_voice_consent(
    State(state): State<AppState>,
    Path(consent_id): Path<String>,
) -> Result<Json<VoiceConsentResource>, ApiError> {
    Ok(Json(consent_or_404(&state, &consent_id).await?.into()))
}

#[derive(Debug, Deserialize)]
pub struct UpdateVoiceConsentRequest {
    name: String,
}

pub async fn update_voice_consent(
    State(state): State<AppState>,
    Path(consent_id): Path<String>,
    ApiJson(req): ApiJson<UpdateVoiceConsentRequest>,
) -> Result<Json<VoiceConsentResource>, ApiError> {
    consent_or_404(&state, &consent_id).await?;
    let consent = state
        .saved_voice_store
        .rename_consent(&consent_id, &req.name)
        .await
        .map_err(store_error)?
        .ok_or_else(|| ApiError::not_found(format!("No voice consent with id {consent_id}")))?;
    Ok(Json(consent.into()))
}

/// Deletes a consent and revokes the voices made with it.
pub async fn delete_voice_consent(
    State(state): State<AppState>,
    Path(consent_id): Path<String>,
) -> Result<Json<DeletedResource>, ApiError> {
    if !state
        .saved_voice_store
        .delete_consent(&consent_id)
        .await
        .map_err(store_error)?
    {
        return Err(ApiError::not_found(format!(
            "No voice consent with id {consent_id}"
        )));
    }
    Ok(Json(DeletedResource {
        id: consent_id,
        object: "audio.voice_consent",
        deleted: true,
        success: None,
    }))
}

#[derive(Debug, Serialize)]
pub struct ConsentPhraseList {
    object: &'static str,
    data: &'static [ConsentPhrase],
}

/// `GET /v1/audio/consent_phrases`: what a `verified` consent recording says.
pub async fn list_consent_phrases() -> Json<ConsentPhraseList> {
    Json(ConsentPhraseList {
        object: "list",
        data: CONSENT_PHRASES,
    })
}

// ------------------------------------------------------------------ voices

/// A voice, as OpenAI's `audio.voice` with SGLang-Omni's fields beside it.
#[derive(Debug, Serialize)]
pub struct VoiceResource {
    object: &'static str,
    id: String,
    name: String,
    #[serde(rename = "type")]
    kind: &'static str,
    created_at: u64,
    /// The consent id, or the attestation given (SGLang-Omni echoes it).
    #[serde(skip_serializing_if = "Option::is_none")]
    consent: Option<String>,
    mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_language: Option<String>,
    /// izwi: whether the voice may still be spoken.
    usable: bool,
}

impl VoiceResource {
    fn from_voice(voice: &SavedVoice, file_size: Option<usize>) -> Self {
        let consent = voice.consent_id.clone().or_else(|| {
            voice
                .permission_provenance
                .as_deref()
                .and_then(|provenance| provenance.strip_prefix("attestation:"))
                .map(str::to_string)
        });
        Self {
            object: "audio.voice",
            id: voice.id.clone(),
            name: voice.name.clone(),
            kind: "audio_sample",
            created_at: unix_secs(voice.created_at),
            consent,
            mime_type: voice.audio_mime_type.clone(),
            file_size,
            ref_text: voice.reference_text.clone(),
            reference_language: voice.reference_language.clone(),
            usable: voice.allows_local_tts_use(),
        }
    }
}

/// How a voice upload's `consent` checks out under the consent mode: the
/// consent it links, and the provenance it records.
async fn check_consent(
    state: &AppState,
    consent: &str,
) -> Result<(Option<String>, String), ApiError> {
    let consent = consent.trim();
    if consent.is_empty() {
        return Err(ApiError::invalid_param(
            "consent",
            "`consent` is required: a consent recording id from /v1/audio/voice_consents, or your statement that the voice is yours or its owner agreed",
        ));
    }
    let recorded = state
        .saved_voice_store
        .get_consent(consent)
        .await
        .map_err(store_error)?;
    match (consent_mode(), recorded) {
        (_, Some(recorded)) if recorded.status == "rejected" => Err(ApiError::invalid_param(
            "consent",
            format!("Consent {} did not match a consent phrase", recorded.id),
        )),
        (VoiceConsentMode::Verified, Some(recorded)) if recorded.status != "verified" => {
            Err(ApiError::invalid_param(
                "consent",
                format!(
                    "Consent {} is not verified; this server needs a recording that says a published consent phrase (GET /v1/audio/consent_phrases)",
                    recorded.id
                ),
            ))
        }
        (_, Some(recorded)) => Ok((Some(recorded.id.clone()), format!("consent:{}", recorded.id))),
        (VoiceConsentMode::Attestation, None) => {
            Ok((None, format!("attestation:{}", consent.chars().take(200).collect::<String>())))
        }
        (_, None) => Err(ApiError::not_found(format!(
            "The consent recording `{consent}` was not found; create one with POST /v1/audio/voice_consents"
        ))),
    }
}

/// `POST /v1/audio/voices`: multipart `name`, `audio_sample` and `consent`
/// (OpenAI), plus `ref_text` and `speaker_description` (SGLang-Omni) and
/// `reference_language` (Izwi). `type=prompt` voices are GPT-Live's; Izwi's
/// equivalent is `/v1/voice-designs`.
pub async fn create_voice(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<VoiceResource>, ApiError> {
    let (mut name, mut consent, mut sample) = (None, None, None::<AudioPayload>);
    let (mut ref_text, mut reference_language, mut kind) = (None, None, None);
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(format!("Invalid multipart payload: {err}")))?
    {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            "name" => name = Some(text_field(field, "name").await?),
            "consent" => consent = Some(text_field(field, "consent").await?),
            "ref_text" | "reference_text" => ref_text = Some(text_field(field, &field_name).await?),
            "reference_language" => {
                reference_language = Some(text_field(field, "reference_language").await?)
            }
            "type" => kind = Some(text_field(field, "type").await?),
            "audio_sample" | "file" => {
                sample = read_multipart_audio_file_payload(
                    field,
                    "audio.voices",
                    "audio_sample",
                    VOICE_UPLOAD_LIMIT_BYTES,
                )
                .await?;
            }
            _ => {}
        }
    }
    if kind.as_deref().is_some_and(|kind| kind != "audio_sample") {
        return Err(ApiError::invalid_param(
            "type",
            "Only `audio_sample` voices can be created here; design a voice from a prompt with /v1/voice-designs.",
        ));
    }
    let name = name
        .filter(|name| !name.is_empty())
        .ok_or_else(|| ApiError::invalid_param("name", "`name` is required"))?;
    if RESERVED_VOICE_NAMES
        .iter()
        .any(|reserved| reserved.eq_ignore_ascii_case(&name))
    {
        return Err(ApiError::invalid_param(
            "name",
            format!("`{name}` is a built-in voice name; choose another"),
        ));
    }
    let sample = sample
        .ok_or_else(|| ApiError::invalid_param("audio_sample", "`audio_sample` is required"))?;
    let (consent_id, provenance) = check_consent(&state, consent.as_deref().unwrap_or("")).await?;
    let inspection = inspect_audio_payload_with_diagnostics("audio.voices", &sample)?;
    crate::api::saved_voices::handlers::validate_saved_voice_reference(inspection.duration_secs)?;
    let reference_language = match reference_language.filter(|value| !value.is_empty()) {
        Some(language) => Some(
            crate::api::saved_voices::handlers::saved_voice_reference_language(Some(&language))?
                .unwrap_or(language),
        ),
        None => None,
    };
    let file_size = sample.bytes.len();
    let voice = state
        .saved_voice_store
        .create_voice(NewSavedVoice {
            name,
            consent_id,
            permission_provenance: Some(provenance),
            reference_text: ref_text.filter(|text| !text.is_empty()),
            reference_language,
            audio_mime_type: sample
                .content_type_hint()
                .unwrap_or("audio/wav")
                .to_string(),
            audio_filename: sample.filename.clone(),
            audio_bytes: sample.bytes,
            source_route_kind: None,
            source_record_id: None,
        })
        .await
        .map_err(store_error)?;
    Ok(Json(VoiceResource::from_voice(&voice, Some(file_size))))
}

#[derive(Debug, Serialize)]
pub struct VoiceList {
    object: &'static str,
    data: Vec<VoiceResource>,
    /// SGLang-Omni: names a request's `voice` may use.
    voices: Vec<String>,
    /// SGLang-Omni: the uploaded voices.
    uploaded_voices: Vec<UploadedVoice>,
}

#[derive(Debug, Serialize)]
pub struct UploadedVoice {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    consent: Option<String>,
    created_at: u64,
    mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    ref_text: Option<String>,
}

/// `GET /v1/audio/voices`: every saved voice, in OpenAI-style `data` and
/// SGLang-Omni's `voices`/`uploaded_voices` (`?names_only=true` gives only
/// `uploaded_voice_names`).
pub async fn list_voices(
    State(state): State<AppState>,
    Query(query): Query<ListQuery>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse;
    let (summaries, _) = state
        .saved_voice_store
        .list_voices_page(query.limit.unwrap_or(500).clamp(1, 2000), None, None)
        .await
        .map_err(store_error)?;
    if query.names_only == Some(true) {
        let names: Vec<String> = summaries.iter().map(|voice| voice.name.clone()).collect();
        return Ok(Json(serde_json::json!({ "uploaded_voice_names": names })).into_response());
    }
    let mut data = Vec::with_capacity(summaries.len());
    for summary in summaries {
        if let Some(voice) = state
            .saved_voice_store
            .get_voice(summary.id.clone())
            .await
            .map_err(store_error)?
        {
            data.push(VoiceResource::from_voice(&voice, None));
        }
    }
    let voices = data.iter().map(|voice| voice.name.clone()).collect();
    let uploaded_voices = data
        .iter()
        .map(|voice| UploadedVoice {
            name: voice.name.clone(),
            consent: voice.consent.clone(),
            created_at: voice.created_at,
            mime_type: voice.mime_type.clone(),
            ref_text: voice.ref_text.clone(),
        })
        .collect();
    Ok(Json(VoiceList {
        object: "list",
        data,
        voices,
        uploaded_voices,
    })
    .into_response())
}

/// The saved voice `id_or_name` names: by id first, then by name.
pub(crate) async fn find_voice(
    state: &AppState,
    id_or_name: &str,
) -> Result<Option<SavedVoice>, ApiError> {
    if let Some(voice) = state
        .saved_voice_store
        .get_voice(id_or_name.trim().to_string())
        .await
        .map_err(store_error)?
    {
        return Ok(Some(voice));
    }
    state
        .saved_voice_store
        .find_voice_by_name(id_or_name)
        .await
        .map_err(store_error)
}

pub async fn get_voice(
    State(state): State<AppState>,
    Path(voice_id): Path<String>,
) -> Result<Json<VoiceResource>, ApiError> {
    let voice = find_voice(&state, &voice_id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("No voice `{voice_id}`")))?;
    Ok(Json(VoiceResource::from_voice(&voice, None)))
}

/// `DELETE /v1/audio/voices/{id_or_name}` (OpenAI deletes by id, SGLang-Omni
/// and vLLM-Omni by name).
pub async fn delete_voice(
    State(state): State<AppState>,
    Extension(_ctx): Extension<RequestContext>,
    Path(voice_id): Path<String>,
) -> Result<Json<DeletedResource>, ApiError> {
    let voice = find_voice(&state, &voice_id)
        .await?
        .ok_or_else(|| ApiError::not_found(format!("No voice `{voice_id}`")))?;
    state
        .saved_voice_store
        .delete_voice(voice.id.clone())
        .await
        .map_err(store_error)?;
    Ok(Json(DeletedResource {
        id: voice.id,
        object: "audio.voice.deleted",
        deleted: true,
        success: Some(true),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consent_phrases_match_by_words_not_punctuation() {
        let phrase = CONSENT_PHRASES[0].text;
        assert_eq!(phrase_match(phrase, phrase), 1.0);
        let said =
            "i am the owner of this voice and i consent to izwi creating a synthetic version of it";
        assert_eq!(phrase_match(said, phrase), 1.0);
        let close =
            "I'm the owner of this voice and I consent to easy creating a synthetic version of it.";
        assert!(phrase_match(close, phrase) >= CONSENT_MATCH_THRESHOLD);
        assert!(phrase_match("hello there, how are you", phrase) < 0.3);
        assert_eq!(phrase_match("anything", ""), 0.0);
    }

    #[test]
    fn reserved_names_cover_openais_voices() {
        assert!(RESERVED_VOICE_NAMES.contains(&"marin"));
        assert_eq!(RESERVED_VOICE_NAMES.len(), 14);
    }
}
