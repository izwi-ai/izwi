//! Custom voices: OpenAI's `audio.output.voice: {"id": ...}` names an Izwi
//! saved voice (`/v1/voices`, or `/v1/audio/voices` where OpenAI SDK code
//! creates them), spoken by a TTS model that clones a reference clip. A
//! voice *name* that is an uploaded voice works too when the profile's TTS
//! model clones and has no built-in voices of its own. A profile whose
//! speaking model cannot clone (built-in voices only, or LFM-Audio) keeps
//! its default voice.

use super::profiles::RealtimeProfile;
use crate::app::voice_realtime::TtsVoice;
use crate::state::AppState;

/// The reference voice saved voice `id` gives the profile's TTS model.
pub(crate) async fn resolve_custom_voice(
    state: &AppState,
    profile: &RealtimeProfile,
    id: &str,
) -> Result<TtsVoice, String> {
    if profile.is_unified() {
        return Err("speech-to-speech models speak only their built-in voices".to_string());
    }
    let model = profile.speech_model();
    let capabilities = model
        .speech_capabilities()
        .ok_or_else(|| format!("`{}` is not a text-to-speech model", model.dir_name()))?;
    if !capabilities.supports_reference_voice {
        return Err(format!(
            "`{}` speaks only its built-in voices; choose a profile whose TTS model clones voices",
            model.dir_name()
        ));
    }
    let reference = crate::api::saved_voices::resolve_saved_voice_reference(state, id)
        .await
        .map_err(|err| err.message)?;
    if capabilities.reference_transcript_required
        && reference
            .reference_text
            .as_deref()
            .is_none_or(|text| text.trim().is_empty())
    {
        return Err(format!(
            "`{}` needs the saved voice's transcript, and voice `{id}` has none",
            model.dir_name()
        ));
    }
    Ok(TtsVoice {
        speaker: None,
        reference_audio: Some(reference.reference_audio_base64),
        reference_text: reference.reference_text,
    })
}

/// The uploaded voice `name` stands for, when the profile's TTS model clones
/// and has no built-in voices (where a name always means a built-in voice).
pub(crate) async fn resolve_named_voice(
    state: &AppState,
    profile: &RealtimeProfile,
    name: &str,
) -> Option<TtsVoice> {
    if profile.is_unified() {
        return None;
    }
    let capabilities = profile.speech_model().speech_capabilities()?;
    if capabilities.supports_builtin_voices || !capabilities.supports_reference_voice {
        return None;
    }
    let voice = state
        .saved_voice_store
        .find_voice_by_name(name)
        .await
        .ok()
        .flatten()?;
    resolve_custom_voice(state, profile, &voice.id).await.ok()
}
