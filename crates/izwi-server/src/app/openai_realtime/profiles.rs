//! What `?model=` means: a realtime profile that is either a modular stack
//! (ASR → chat → TTS) or a unified speech-to-speech model (LFM-Audio), plus
//! the mapping of OpenAI voice and transcription model names onto Izwi's.
//!
//! Resolution order for `model`:
//! 1. a profile name: the built-in `izwi-voice` (modular, the desktop's
//!    default stack) and `izwi-voice-compact` (LFM3-Audio), or one defined in
//!    `IZWI_REALTIME_PROFILES`;
//! 2. an audio-chat model id: unified with that model;
//! 3. a chat model id: modular with that text model and the default ASR/TTS;
//! 4. an OpenAI realtime model id (`gpt-realtime*`, `gpt-4o*-realtime*`,
//!    `gpt-audio*`), absent or empty: the default profile
//!    (`IZWI_REALTIME_DEFAULT_PROFILE`, `izwi-voice` unless set), so stock
//!    clients work unchanged.

use izwi_core::{
    parse_chat_model_variant, parse_model_variant, parse_tts_model_variant, ModelVariant,
};
use serde::{Deserialize, Serialize};

use crate::app::voice_realtime::{validate_realtime_text_model, validate_realtime_tts_model};

pub(crate) const DEFAULT_PROFILE_ID: &str = "izwi-voice";
pub(crate) const COMPACT_PROFILE_ID: &str = "izwi-voice-compact";
/// The desktop's default modular stack (`ui/src/features/voice/realtime/support.ts`).
pub(crate) const DEFAULT_ASR_MODEL: &str = "Parakeet-TDT-0.6B-v3";
pub(crate) const DEFAULT_TEXT_MODEL: &str = "Qwen3.5-4B";
pub(crate) const DEFAULT_TTS_MODEL: &str = "Kokoro-82M";
/// The desktop's compact voice model (`ui/src/features/voice/engine.ts`).
pub(crate) const DEFAULT_S2S_MODEL: &str = "LFM3-Audio-2B-GGUF";
/// The voice a session starts with, as OpenAI's own examples do.
pub(crate) const DEFAULT_VOICE: &str = "marin";

/// One realtime profile as configured.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ProfileSpec {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) asr_model_id: Option<String>,
    #[serde(default)]
    pub(crate) text_model_id: Option<String>,
    #[serde(default)]
    pub(crate) tts_model_id: Option<String>,
    #[serde(default)]
    pub(crate) s2s_model_id: Option<String>,
    #[serde(default)]
    pub(crate) voice: Option<String>,
}

/// A resolved profile: its models, validated for realtime use.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RealtimeProfile {
    pub(crate) id: String,
    pub(crate) mode: ProfileMode,
    pub(crate) default_voice: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProfileMode {
    Modular {
        asr_model_id: String,
        asr: ModelVariant,
        text_model_id: String,
        text: ModelVariant,
        tts_model_id: String,
        tts: ModelVariant,
    },
    Unified {
        s2s_model_id: String,
        s2s: ModelVariant,
    },
}

impl RealtimeProfile {
    pub(crate) fn is_unified(&self) -> bool {
        matches!(self.mode, ProfileMode::Unified { .. })
    }

    /// The model that speaks: the TTS model, or the unified model.
    pub(crate) fn speech_model(&self) -> ModelVariant {
        match &self.mode {
            ProfileMode::Modular { tts, .. } => *tts,
            ProfileMode::Unified { s2s, .. } => *s2s,
        }
    }

    /// The profile's models in pipeline order: (stage, id, variant).
    pub(crate) fn models(&self) -> Vec<(&'static str, String, ModelVariant)> {
        match &self.mode {
            ProfileMode::Modular {
                asr_model_id,
                asr,
                text_model_id,
                text,
                tts_model_id,
                tts,
            } => vec![
                ("asr", asr_model_id.clone(), *asr),
                ("text", text_model_id.clone(), *text),
                ("tts", tts_model_id.clone(), *tts),
            ],
            ProfileMode::Unified { s2s_model_id, s2s } => {
                vec![("s2s", s2s_model_id.clone(), *s2s)]
            }
        }
    }

    pub(crate) fn summary(&self) -> String {
        self.models()
            .iter()
            .map(|(stage, id, _)| format!("{stage}={id}"))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn builtin_specs() -> Vec<ProfileSpec> {
    vec![
        ProfileSpec {
            id: DEFAULT_PROFILE_ID.to_string(),
            asr_model_id: Some(DEFAULT_ASR_MODEL.to_string()),
            text_model_id: Some(DEFAULT_TEXT_MODEL.to_string()),
            tts_model_id: Some(DEFAULT_TTS_MODEL.to_string()),
            s2s_model_id: None,
            voice: None,
        },
        ProfileSpec {
            id: COMPACT_PROFILE_ID.to_string(),
            asr_model_id: None,
            text_model_id: None,
            tts_model_id: None,
            s2s_model_id: Some(DEFAULT_S2S_MODEL.to_string()),
            voice: None,
        },
    ]
}

/// The profiles configured: `IZWI_REALTIME_PROFILES` (a JSON array of
/// `{id, asr_model_id, text_model_id, tts_model_id | s2s_model_id, voice}`),
/// then the built-ins it does not redefine.
pub(crate) fn configured_specs() -> Result<Vec<ProfileSpec>, String> {
    let mut specs: Vec<ProfileSpec> = match std::env::var("IZWI_REALTIME_PROFILES") {
        Ok(raw) if !raw.trim().is_empty() => serde_json::from_str(&raw).map_err(|err| {
            format!("IZWI_REALTIME_PROFILES is not a JSON array of profiles: {err}")
        })?,
        _ => Vec::new(),
    };
    for builtin in builtin_specs() {
        if !specs
            .iter()
            .any(|spec| spec.id.eq_ignore_ascii_case(&builtin.id))
        {
            specs.push(builtin);
        }
    }
    Ok(specs)
}

fn default_profile_id() -> String {
    std::env::var("IZWI_REALTIME_DEFAULT_PROFILE")
        .ok()
        .map(|id| id.trim().to_string())
        .filter(|id| !id.is_empty())
        .unwrap_or_else(|| DEFAULT_PROFILE_ID.to_string())
}

/// Whether `model` names one of OpenAI's realtime models.
pub(crate) fn is_openai_realtime_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("gpt-realtime")
        || model.starts_with("gpt-4o-realtime")
        || model.starts_with("gpt-4o-mini-realtime")
        || model.starts_with("gpt-audio")
}

/// Validate a profile's models for realtime use.
pub(crate) fn validate_spec(spec: &ProfileSpec) -> Result<RealtimeProfile, String> {
    let default_voice = spec
        .voice
        .clone()
        .unwrap_or_else(|| DEFAULT_VOICE.to_string());
    if let Some(s2s_model_id) = spec.s2s_model_id.as_deref() {
        let s2s = parse_model_variant(s2s_model_id)
            .map_err(|err| format!("Profile `{}`: unsupported model: {err}", spec.id))?;
        if !s2s.is_audio_chat() {
            return Err(format!(
                "Profile `{}`: `{s2s_model_id}` is not a speech-to-speech (audio chat) model",
                spec.id
            ));
        }
        return Ok(RealtimeProfile {
            id: spec.id.clone(),
            mode: ProfileMode::Unified {
                s2s_model_id: s2s.dir_name().to_string(),
                s2s,
            },
            default_voice,
        });
    }
    let asr_model_id = spec.asr_model_id.as_deref().unwrap_or(DEFAULT_ASR_MODEL);
    let text_model_id = spec.text_model_id.as_deref().unwrap_or(DEFAULT_TEXT_MODEL);
    let tts_model_id = spec.tts_model_id.as_deref().unwrap_or(DEFAULT_TTS_MODEL);
    let asr = parse_model_variant(asr_model_id)
        .map_err(|err| format!("Profile `{}`: unsupported ASR model: {err}", spec.id))?;
    if !transcribes_input(asr) {
        return Err(format!(
            "Profile `{}`: `{asr_model_id}` is not a live ASR model",
            spec.id
        ));
    }
    let text = parse_chat_model_variant(Some(text_model_id))
        .map_err(|err| format!("Profile `{}`: invalid chat model: {err}", spec.id))?;
    validate_realtime_text_model(text).map_err(|err| format!("Profile `{}`: {err}", spec.id))?;
    let tts = parse_tts_model_variant(tts_model_id)
        .map_err(|err| format!("Profile `{}`: unsupported TTS model: {err}", spec.id))?;
    validate_realtime_tts_model(tts).map_err(|err| format!("Profile `{}`: {err}", spec.id))?;
    Ok(RealtimeProfile {
        id: spec.id.clone(),
        mode: ProfileMode::Modular {
            asr_model_id: asr.dir_name().to_string(),
            asr,
            text_model_id: text.dir_name().to_string(),
            text,
            tts_model_id: tts.dir_name().to_string(),
            tts,
        },
        default_voice,
    })
}

fn spec_by_id(specs: &[ProfileSpec], id: &str) -> Option<ProfileSpec> {
    specs
        .iter()
        .find(|spec| spec.id.eq_ignore_ascii_case(id.trim()))
        .cloned()
}

/// The profile `model` selects (see the module docs). An unknown model is
/// an error for the upgrade to refuse.
pub(crate) fn resolve_profile(model: Option<&str>) -> Result<RealtimeProfile, String> {
    let specs = configured_specs()?;
    let requested = model.map(str::trim).filter(|model| !model.is_empty());
    let Some(requested) = requested.filter(|model| !is_openai_realtime_model(model)) else {
        let default_id = default_profile_id();
        let spec = spec_by_id(&specs, &default_id).ok_or_else(|| {
            format!("IZWI_REALTIME_DEFAULT_PROFILE names no profile: `{default_id}`")
        })?;
        return validate_spec(&spec);
    };
    if let Some(spec) = spec_by_id(&specs, requested) {
        return validate_spec(&spec);
    }
    let variant = parse_model_variant(requested).map_err(|_| unknown_model(requested, &specs))?;
    let spec = if variant.is_audio_chat() {
        ProfileSpec {
            id: requested.to_string(),
            s2s_model_id: Some(requested.to_string()),
            ..empty_spec()
        }
    } else if variant.is_chat() {
        ProfileSpec {
            id: requested.to_string(),
            text_model_id: Some(requested.to_string()),
            ..empty_spec()
        }
    } else {
        return Err(unknown_model(requested, &specs));
    };
    validate_spec(&spec)
}

fn empty_spec() -> ProfileSpec {
    ProfileSpec {
        id: String::new(),
        asr_model_id: None,
        text_model_id: None,
        tts_model_id: None,
        s2s_model_id: None,
        voice: None,
    }
}

fn unknown_model(requested: &str, specs: &[ProfileSpec]) -> String {
    let names = specs
        .iter()
        .map(|spec| spec.id.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Unknown realtime model `{requested}`. Use a realtime profile ({names}), a speech-to-speech \
         model id, a chat model id, or an OpenAI realtime model name for the default profile."
    )
}

/// OpenAI voices and whether they sound female (`true`) or male. `fable`,
/// `nova` and `onyx` are the speech API's and come along for old clients.
const OPENAI_VOICES: &[(&str, bool)] = &[
    ("alloy", true),
    ("ash", false),
    ("ballad", false),
    ("cedar", false),
    ("coral", true),
    ("echo", false),
    ("fable", false),
    ("marin", true),
    ("nova", true),
    ("onyx", false),
    ("sage", true),
    ("shimmer", true),
    ("verse", false),
];

fn openai_voice_is_female(voice: &str) -> Option<(usize, bool)> {
    OPENAI_VOICES
        .iter()
        .position(|(name, _)| name.eq_ignore_ascii_case(voice.trim()))
        .map(|index| (index, OPENAI_VOICES[index].1))
}

/// The speaker `voice` asks of the model that speaks. OpenAI names map onto
/// the model's own voices (Kokoro, Maya1 and Maya2 resolve OpenAI names
/// themselves); any other name is the model's own and passes through.
/// `None` lets the model use its default.
pub(crate) fn speaker_for(speech_model: ModelVariant, voice: &str) -> Option<String> {
    use izwi_core::ModelFamily;
    let voice = voice.trim();
    if voice.is_empty() {
        return None;
    }
    let Some((index, female)) = openai_voice_is_female(voice) else {
        return Some(voice.to_string());
    };
    let pick = |voices: &[&str]| voices[index % voices.len()].to_string();
    match speech_model.family() {
        ModelFamily::KokoroTts | ModelFamily::Maya1Tts | ModelFamily::Maya2Tts => {
            Some(voice.to_ascii_lowercase())
        }
        ModelFamily::Qwen3Tts => Some(if female {
            pick(&["Vivian", "Serena", "Sohee"])
        } else {
            pick(&["Ryan", "Aiden", "Dylan", "Eric"])
        }),
        ModelFamily::VoxtralTts => Some(if female {
            pick(&["neutral_female", "casual_female", "cheerful_female"])
        } else {
            pick(&["neutral_male", "casual_male"])
        }),
        ModelFamily::Lfm3Audio => Some(if female { "Female" } else { "Male" }.to_string()),
        ModelFamily::Lfm25Audio => Some(if female { "US Female" } else { "US Male" }.to_string()),
        // One bundled voice (Chatterbox), or none to map onto.
        _ => None,
    }
}

/// Whether `model` (an `audio.input.transcription.model`) is one of
/// OpenAI's transcription models, which run on the profile's own ASR.
pub(crate) fn is_openai_transcription_model(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model == "whisper-1"
        || model.starts_with("gpt-4o-transcribe")
        || model.starts_with("gpt-4o-mini-transcribe")
        || model.starts_with("gpt-transcribe")
        || model.starts_with("gpt-live-transcribe")
        || model.starts_with("gpt-realtime-whisper")
}

/// Whether `variant` may transcribe a session's input: a live ASR model, or
/// a model with a streaming session of its own (Voxtral Realtime,
/// LFM3-Audio).
pub(crate) fn transcribes_input(variant: ModelVariant) -> bool {
    variant.serves_live_asr() || izwi_core::asr_streams_live(variant)
}

/// The ASR model that transcribes a session's input: the transcription
/// model the client named when it is an Izwi live ASR model (after the
/// operator's `IZWI_OPENAI_MODEL_ALIASES`, so `gpt-4o-transcribe=…` picks
/// that model), else the profile's (a modular stack's ASR, LFM3-Audio
/// itself, or the default ASR beside LFM2.5-Audio, which has no turn
/// transcript).
pub(crate) fn transcription_model(
    profile: &RealtimeProfile,
    requested: Option<&str>,
) -> Result<String, String> {
    transcription_model_with(profile, requested, |model| {
        crate::api::openai::model_aliases::resolve_model_alias(model).to_string()
    })
}

fn transcription_model_with(
    profile: &RealtimeProfile,
    requested: Option<&str>,
    resolve_alias: impl Fn(&str) -> String,
) -> Result<String, String> {
    let requested = requested
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .map(resolve_alias);
    if let Some(requested) = requested
        .as_deref()
        .filter(|model| !is_openai_transcription_model(model))
    {
        let variant = parse_model_variant(requested)
            .map_err(|_| format!("Unknown transcription model `{requested}`"))?;
        if !transcribes_input(variant) {
            return Err(format!(
                "`{requested}` cannot transcribe live speech; name an ASR model"
            ));
        }
        return Ok(variant.dir_name().to_string());
    }
    Ok(match &profile.mode {
        ProfileMode::Modular { asr_model_id, .. } => asr_model_id.clone(),
        ProfileMode::Unified { s2s, s2s_model_id }
            if s2s.family() == izwi_core::ModelFamily::Lfm3Audio =>
        {
            s2s_model_id.clone()
        }
        ProfileMode::Unified { .. } => DEFAULT_ASR_MODEL.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_model_names_and_empty_models_take_the_default_profile() {
        for model in [
            None,
            Some(""),
            Some("gpt-realtime"),
            Some("gpt-realtime-2.1"),
            Some("gpt-4o-realtime-preview-2025-06-03"),
            Some("gpt-audio-mini"),
        ] {
            let profile = resolve_profile(model).expect("default profile");
            assert_eq!(profile.id, DEFAULT_PROFILE_ID, "{model:?}");
            assert!(!profile.is_unified());
        }
    }

    #[test]
    fn model_ids_select_modular_or_unified_stacks() {
        let compact = resolve_profile(Some("izwi-voice-compact")).expect("compact");
        assert!(compact.is_unified());
        let lfm = resolve_profile(Some("LFM2.5-Audio-1.5B-GGUF")).expect("lfm2.5");
        assert!(lfm.is_unified());
        let chat = resolve_profile(Some("Qwen3.5-0.8B")).expect("chat");
        match chat.mode {
            ProfileMode::Modular {
                text_model_id,
                tts_model_id,
                ..
            } => {
                assert_eq!(text_model_id, "Qwen3.5-0.8B");
                assert_eq!(tts_model_id, "Kokoro-82M");
            }
            other => panic!("modular expected, got {other:?}"),
        }
        assert!(
            resolve_profile(Some("Kokoro-82M")).is_err(),
            "a TTS model is no stack"
        );
        let err = resolve_profile(Some("no-such-model")).expect_err("unknown");
        assert!(err.contains("izwi-voice"), "{err}");
    }

    #[test]
    fn openai_voices_map_onto_each_models_voices() {
        let kokoro = parse_tts_model_variant("Kokoro-82M").expect("kokoro");
        assert_eq!(speaker_for(kokoro, "Marin").as_deref(), Some("marin"));
        assert_eq!(speaker_for(kokoro, "af_heart").as_deref(), Some("af_heart"));
        let lfm3 = parse_model_variant("LFM3-Audio-2B-GGUF").expect("lfm3");
        assert_eq!(speaker_for(lfm3, "marin").as_deref(), Some("Female"));
        assert_eq!(speaker_for(lfm3, "cedar").as_deref(), Some("Male"));
        let lfm25 = parse_model_variant("LFM2.5-Audio-1.5B-GGUF").expect("lfm2.5");
        assert_eq!(speaker_for(lfm25, "ash").as_deref(), Some("US Male"));
        assert_eq!(speaker_for(lfm3, "  "), None);
        assert_eq!(
            openai_voice_is_female("SHIMMER").map(|(_, female)| female),
            Some(true)
        );
        assert_eq!(openai_voice_is_female("af_heart"), None);
    }

    #[test]
    fn transcription_runs_on_the_profiles_asr_unless_an_izwi_asr_is_named() {
        let modular = resolve_profile(None).expect("default");
        assert_eq!(
            transcription_model(&modular, Some("gpt-4o-mini-transcribe")).expect("openai"),
            DEFAULT_ASR_MODEL
        );
        assert_eq!(
            transcription_model(&modular, Some("whisper-1")).expect("whisper-1"),
            DEFAULT_ASR_MODEL
        );
        let compact = resolve_profile(Some(COMPACT_PROFILE_ID)).expect("compact");
        assert_eq!(
            transcription_model(&compact, None).expect("lfm3"),
            DEFAULT_S2S_MODEL
        );
        let lfm25 = resolve_profile(Some("LFM2.5-Audio-1.5B-GGUF")).expect("lfm2.5");
        assert_eq!(
            transcription_model(&lfm25, None).expect("asr"),
            DEFAULT_ASR_MODEL
        );
        assert!(transcription_model(&modular, Some("Kokoro-82M")).is_err());
    }

    /// An operator alias for an OpenAI transcription name picks that model
    /// instead of the profile's ASR.
    #[test]
    fn transcription_aliases_pick_the_mapped_asr() {
        let modular = resolve_profile(None).expect("default");
        let alias = |model: &str| {
            if model == "gpt-4o-transcribe" {
                "Parakeet-TDT-0.6B-v3".to_string()
            } else {
                model.to_string()
            }
        };
        let mapped =
            transcription_model_with(&modular, Some("gpt-4o-transcribe"), alias).expect("mapped");
        assert_eq!(
            parse_model_variant(&mapped).expect("variant"),
            parse_model_variant("Parakeet-TDT-0.6B-v3").expect("parakeet")
        );
        assert_eq!(
            transcription_model_with(&modular, Some("whisper-1"), alias).expect("unmapped"),
            DEFAULT_ASR_MODEL
        );
    }
}
