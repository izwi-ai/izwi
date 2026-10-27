use izwi_core::catalog::ModelFamily;
use izwi_core::runtime_models::architectures::chatterbox::{
    chatterbox_auto_max_speech_tokens, ChatterboxEdition,
};
pub(crate) use izwi_core::runtime_models::architectures::qwen3::tts::{
    qwen3_tts_auto_max_frames_for_text as qwen_tts_auto_max_frames_for_text,
};
use izwi_core::runtime_models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames;
use izwi_core::runtime_models::architectures::vibevoice::tts::{
    vibevoice_tts_auto_max_frames_for_text, vibevoice_tts_auto_max_output_frames,
};
use izwi_core::runtime_models::architectures::voxtral::tts::voxtral_tts_auto_max_frames_for_text;
use izwi_core::ModelVariant;

/// The longest speech input one request may carry, in UTF-8 bytes
/// (`IZWI_TTS_MAX_TEXT_BYTES`, default 1 MiB). Long form splits text into
/// model-sized calls, so this bounds only the request itself (TPIPE-14).
pub(crate) fn tts_max_text_bytes() -> usize {
    crate::env_config::tuning().tts_max_text_bytes
}

/// The voice a speech request asks for, as the routes read it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SpeechVoiceRequest {
    /// A built-in speaker was named.
    pub(crate) speaker: bool,
    pub(crate) reference_audio: bool,
    pub(crate) reference_text: bool,
    /// The reference came from a saved voice (`saved_voice_id`).
    pub(crate) saved_voice: bool,
    /// A voice description (VoiceDesign `instructions`).
    pub(crate) voice_description: bool,
}

/// The one capability-driven voice contract every speech route applies
/// (TPIPE-12): a reference is its audio, with its transcript for every model
/// that conditions on the reference's words (all but Chatterbox); a model
/// is given only the voice inputs it supports; and a model whose only voice
/// mode is a reference (Qwen Base, VibeVoice, Fish) or a description (Qwen
/// VoiceDesign) must be given one.
pub(crate) fn validate_speech_voice(
    variant: ModelVariant,
    voice: SpeechVoiceRequest,
) -> Result<(), crate::error::ApiError> {
    use crate::error::ApiError;
    let capabilities = variant.speech_capabilities().ok_or_else(|| {
        ApiError::bad_request(format!("{variant} is not a text-to-speech model."))
    })?;
    if voice.reference_text && !voice.reference_audio {
        return Err(ApiError::bad_request(
            "`reference_text` was given without `reference_audio`.",
        ));
    }
    let reference = voice.reference_audio;
    if voice.speaker && !capabilities.supports_builtin_voices {
        return Err(ApiError::bad_request(format!(
            "{variant} does not support built-in speaker selection.",
        )));
    }
    if reference && !capabilities.supports_reference_voice {
        return Err(ApiError::bad_request(format!(
            "{variant} does not support reference or saved voices.",
        )));
    }
    if reference && capabilities.reference_transcript_required && !voice.reference_text {
        return Err(ApiError::bad_request(if voice.saved_voice {
            format!(
                "This saved voice has no transcript, which {variant} needs. Use a voice saved \
                 with one, or a model that clones from audio alone."
            )
        } else {
            "Provide both `reference_audio` and `reference_text` together.".to_string()
        }));
    }
    if voice.voice_description && !capabilities.supports_voice_description {
        return Err(ApiError::bad_request(format!(
            "{variant} does not support voice direction prompts.",
        )));
    }
    if !capabilities.supports_builtin_voices
        && !capabilities.supports_reference_voice
        && !capabilities.supports_voice_description
    {
        return Err(ApiError::bad_request(format!(
            "{variant} does not expose a supported text-to-speech voice mode.",
        )));
    }
    if capabilities.supports_reference_voice
        && !capabilities.supports_builtin_voices
        && !capabilities.supports_voice_description
        && !reference
    {
        let message = if capabilities.reference_transcript_required {
            format!(
                "{variant} requires `saved_voice_id` or both `reference_audio` and `reference_text`."
            )
        } else {
            format!("{variant} requires `saved_voice_id` or `reference_audio`.")
        };
        return Err(ApiError::bad_request(message));
    }
    if capabilities.supports_voice_description
        && !capabilities.supports_builtin_voices
        && !capabilities.supports_reference_voice
        && !voice.voice_description
    {
        return Err(ApiError::bad_request(format!(
            "{variant} requires a voice description (`instructions` or `voice_description`).",
        )));
    }
    Ok(())
}

/// Refuse a `speed` the model cannot honour with 400 (TPIPE-10): any speed
/// but 1.0 on a model without speed control, or one outside the supported
/// range, instead of clamping or dropping it silently.
pub(crate) fn validate_speech_speed(
    variant: ModelVariant,
    speed: Option<f32>,
) -> Result<(), crate::error::ApiError> {
    let Some(speed) = speed else {
        return Ok(());
    };
    let capabilities = variant.speech_capabilities();
    if capabilities
        .as_ref()
        .is_some_and(|capabilities| capabilities.honours_speed(speed))
        || (capabilities.is_none() && speed == 1.0)
    {
        return Ok(());
    }
    let (low, high) = izwi_core::catalog::SpeechModelCapabilities::SPEED_RANGE;
    Err(crate::error::ApiError::bad_request(
        if capabilities.is_some_and(|capabilities| capabilities.supports_speed_control) {
            format!("`speed` must be between {low} and {high} for {variant}; got {speed}")
        } else {
            format!("{variant} does not support speed control; omit `speed` or set it to 1.0")
        },
    ))
}

/// Refuse `stream` for a model that only produces a finished utterance,
/// matching the native speech routes, instead of failing inside the
/// runtime.
pub(crate) fn validate_speech_streaming(
    variant: ModelVariant,
    streaming: bool,
) -> Result<(), crate::error::ApiError> {
    if streaming
        && variant
            .speech_capabilities()
            .is_some_and(|capabilities| !capabilities.supports_streaming)
    {
        return Err(crate::error::ApiError::bad_request(format!(
            "{variant} does not support streaming synthesis. Disable `stream` or choose a model with streaming support."
        )));
    }
    Ok(())
}

/// Refuse a model-specific control (`exaggeration`, `cfg_weight`,
/// `min_p`) the model does not take, or one outside its range, the way
/// `speed` is refused, instead of ignoring it.
pub(crate) fn validate_speech_controls(
    variant: ModelVariant,
    sampling: &SpeechSamplingRequest,
) -> Result<(), crate::error::ApiError> {
    use izwi_core::catalog::SpeechControl;
    for (control, value) in [
        (SpeechControl::Exaggeration, sampling.exaggeration),
        (SpeechControl::CfgWeight, sampling.cfg_weight),
        (SpeechControl::MinP, sampling.min_p),
    ] {
        let Some(value) = value else {
            continue;
        };
        let name = control.as_str();
        let Some(range) = variant
            .speech_controls()
            .iter()
            .find(|range| range.control == control)
        else {
            return Err(crate::error::ApiError::bad_request(format!(
                "{variant} does not support `{name}`; omit it"
            )));
        };
        if !value.is_finite() || value < range.min || value > range.max {
            return Err(crate::error::ApiError::bad_request(format!(
                "`{name}` must be between {} and {} for {variant}; got {value}",
                range.min, range.max
            )));
        }
    }
    Ok(())
}

/// Refuse a `language` the model cannot speak before it loads: the English
/// Chatterbox takes English only, the multilingual one its supported codes.
/// Other models read `language` as a hint.
pub(crate) fn validate_speech_language(
    variant: ModelVariant,
    language: Option<&str>,
) -> Result<(), crate::error::ApiError> {
    let Some(edition) = ChatterboxEdition::of(variant) else {
        return Ok(());
    };
    match edition.language_tag(language) {
        Ok(_) => Ok(()),
        Err(izwi_core::Error::InvalidInput(message)) => {
            Err(crate::error::ApiError::bad_request(message))
        }
        Err(err) => Err(crate::error::core_api_error(err)),
    }
}

/// Refuse an over-limit speech input with 413 before any model loads.
pub(crate) fn check_tts_input_length(text: &str) -> Result<(), crate::error::ApiError> {
    let limit = tts_max_text_bytes();
    if text.len() > limit {
        return Err(crate::error::ApiError::payload_too_large(format!(
            "Speech input is {} bytes, above the {limit}-byte limit; send it as several requests",
            text.len()
        )));
    }
    Ok(())
}

/// Optional speech-sampling fields a request may set beside `temperature`
/// and `top_k`. Anything left unset keeps the model family's reference
/// default (TPIPE-24): Qwen3-TTS generation_config, LFM 0.8 / top-k 64,
/// VibeVoice CFG 1.3 with 10 steps, fish-speech 0.7 / 0.7, Voxtral greedy.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct SpeechSamplingRequest {
    /// Nucleus sampling threshold, in (0, 1].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Sampling seed (Qwen3-TTS, LFM2.5-Audio, Fish S2, Voxtral noise, Kokoro noise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    /// Repetition penalty (Qwen3-TTS), >= 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    /// Qwen3-TTS residual-codebook sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_do_sample: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtalker_top_p: Option<f32>,
    /// VibeVoice classifier-free guidance scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
    /// VibeVoice diffusion steps per frame.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion_steps: Option<usize>,
    /// Speak numbers, currency, dates, times and units as words (English).
    /// Unset keeps the model's default: on for Kokoro, off for models that
    /// read them themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalize_text: Option<bool>,
    /// Min-p sampling threshold, in [0, 1] (Chatterbox).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    /// Emotion intensity, in [0, 2] (Chatterbox; its default is 0.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exaggeration: Option<f32>,
    /// Classifier-free guidance weight, in [0, 1]; 0 turns guidance off
    /// (Chatterbox; its default is 0.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg_weight: Option<f32>,
}

impl SpeechSamplingRequest {
    /// The engine's speech sampling for a request that also set `temperature`
    /// and `top_k` (each `None` when the client left it unset).
    pub(crate) fn options(
        &self,
        temperature: Option<f32>,
        top_k: Option<usize>,
    ) -> izwi_core::SpeechSamplingOptions {
        izwi_core::SpeechSamplingOptions {
            temperature,
            top_p: self.top_p,
            top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
            subtalker_do_sample: self.subtalker_do_sample,
            subtalker_temperature: self.subtalker_temperature,
            subtalker_top_k: self.subtalker_top_k,
            subtalker_top_p: self.subtalker_top_p,
            cfg_scale: self.cfg_scale,
            diffusion_steps: self.diffusion_steps,
            normalize_text: self.normalize_text,
            min_p: self.min_p,
            exaggeration: self.exaggeration,
            cfg_weight: self.cfg_weight,
        }
    }
}

/// The largest explicit output-frame budget any backend honours for
/// `variant`.
fn explicit_frame_ceiling(variant: ModelVariant) -> usize {
    let hint = variant.tts_max_output_frames_hint().unwrap_or(1);
    if variant == ModelVariant::Voxtral4BTts2603 {
        hint.max(ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES)
    } else {
        hint
    }
}

pub(crate) fn resolve_tts_output_frames(
    variant: ModelVariant,
    text: &str,
    requested_frames: Option<usize>,
) -> Option<usize> {
    let model_max_frames = variant.tts_max_output_frames_hint()?;
    match requested_frames {
        // An explicit budget is clamped to the largest any backend honours;
        // the runtime clamps it again to the loaded backend's limit (Voxtral
        // allows 2,048 frames on CUDA, not the 1,500-frame hint; TPIPE-19).
        Some(value) if value > 0 => Some(value.clamp(1, explicit_frame_ceiling(variant))),
        Some(0) | None if variant == ModelVariant::Voxtral4BTts2603 => {
            Some(voxtral_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::VibeVoice15BTts => {
            Some(vibevoice_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant == ModelVariant::FishAudioS2Pro => {
            Some(fish_s2_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::ChatterboxTts => {
            Some(chatterbox_auto_max_speech_tokens(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::Qwen3Tts => {
            Some(qwen_tts_auto_max_frames_for_text(text).min(model_max_frames))
        }
        Some(0) | None => Some(model_max_frames),
        Some(_) => unreachable!("positive requested frames handled above"),
    }
}

/// Whether `text` is too long for one automatic request of `variant`: its
/// automatic budget reaches the per-request cap, so one request would cut
/// it. Only families that advertise automatic long form split; Fish plans
/// its own segments.
pub(crate) fn tts_text_exceeds_one_request(variant: ModelVariant, text: &str) -> bool {
    if !variant
        .speech_capabilities()
        .is_some_and(|capabilities| capabilities.supports_auto_long_form)
    {
        return false;
    }
    let cap = match variant.family() {
        ModelFamily::Qwen3Tts => qwen3_tts_auto_max_output_frames(),
        ModelFamily::VibeVoiceTts => vibevoice_tts_auto_max_output_frames(),
        ModelFamily::ChatterboxTts => ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES,
        _ => return false,
    };
    resolve_tts_output_frames(variant, text, None).is_some_and(|frames| frames >= cap)
}

fn fish_s2_tts_auto_max_frames_for_text(text: &str) -> usize {
    // Whitespace alone treats an entire CJK paragraph as one word. Include a
    // character estimate and count CJK syllabic characters separately; leave
    // time for pauses and EOS rather than truncating at the speaking estimate.
    let non_space = text.chars().filter(|ch| !ch.is_whitespace()).count();
    let cjk = text
        .chars()
        .filter(|ch| {
            matches!(*ch as u32,
                0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff |
                0xac00..=0xd7af | 0x20000..=0x3134f
            )
        })
        .count();
    let estimated_words = text
        .split_whitespace()
        .count()
        .max(non_space.saturating_sub(cjk).div_ceil(5));
    // This is a per-segment ceiling, not a total-recording limit. Budget for
    // slower narration and pauses; the model's EOS ends normal speech earlier.
    let estimated_secs = (estimated_words as f32 / 1.8 + cjk as f32 / 2.5 + 4.0).clamp(6.0, 120.0);
    let frames = (estimated_secs * ModelVariant::FISH_S2_PRO_FRAME_RATE_HZ).ceil() as usize;
    frames.clamp(96, ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_transcript_is_required_only_by_models_that_condition_on_it() {
        let voice = |reference_audio, reference_text| SpeechVoiceRequest {
            reference_audio,
            reference_text,
            ..Default::default()
        };
        let chatterbox = ModelVariant::Chatterbox500M;
        validate_speech_voice(chatterbox, voice(true, false)).expect("audio alone clones");
        let saved_without_transcript = SpeechVoiceRequest {
            reference_audio: true,
            saved_voice: true,
            ..Default::default()
        };
        validate_speech_voice(chatterbox, saved_without_transcript)
            .expect("a saved voice without a transcript serves Chatterbox");
        let err = validate_speech_voice(ModelVariant::FishAudioS2Pro, saved_without_transcript)
            .unwrap_err();
        assert!(err.message.contains("saved voice has no transcript"));
        validate_speech_voice(chatterbox, voice(true, true)).expect("a transcript is ignored");
        let err = validate_speech_voice(chatterbox, voice(false, true)).unwrap_err();
        assert!(err.message.contains("without `reference_audio`"));

        for variant in [ModelVariant::FishAudioS2Pro, ModelVariant::VibeVoice15BTts] {
            let err = validate_speech_voice(variant, voice(true, false)).unwrap_err();
            assert!(err.message.contains("Provide both"), "{variant}");
            validate_speech_voice(variant, voice(true, true)).expect("audio and transcript");
            let err = validate_speech_voice(variant, voice(false, false)).unwrap_err();
            assert!(err
                .message
                .contains("both `reference_audio` and `reference_text`"));
        }
    }

    /// TPIPE-19: an explicit Voxtral budget up to CUDA's 2,048 frames
    /// reaches the runtime (which clamps it to the loaded backend's limit),
    /// rather than being cut to the 1,500-frame hint here.
    #[test]
    fn explicit_budgets_reach_the_runtime_up_to_the_largest_backend_limit() {
        let voxtral = ModelVariant::Voxtral4BTts2603;
        assert_eq!(
            resolve_tts_output_frames(voxtral, "hi", Some(2_000)),
            Some(2_000)
        );
        assert_eq!(
            resolve_tts_output_frames(voxtral, "hi", Some(9_999)),
            Some(ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES)
        );
        let qwen = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let qwen_cap = qwen.tts_max_output_frames_hint().unwrap();
        assert_eq!(
            resolve_tts_output_frames(qwen, "hi", Some(usize::MAX)),
            Some(qwen_cap)
        );
    }

    /// TPIPE-10: Qwen3-TTS reads no `speed`, so it no longer advertises
    /// speed control and refuses any speed but 1.0; Kokoro honours 0.5-2.0
    /// and refuses a speed it would have clamped.
    #[test]
    fn speed_is_honoured_or_refused_never_clamped_or_dropped() {
        let qwen = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let kokoro = ModelVariant::Kokoro82M;
        assert!(validate_speech_speed(qwen, None).is_ok());
        assert!(validate_speech_speed(qwen, Some(1.0)).is_ok());
        let error = validate_speech_speed(qwen, Some(1.5)).unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message.contains("does not support speed control"));
        assert!(validate_speech_speed(kokoro, Some(0.5)).is_ok());
        assert!(validate_speech_speed(kokoro, Some(2.0)).is_ok());
        let error = validate_speech_speed(kokoro, Some(0.25)).unwrap_err();
        assert!(
            error.message.contains("between 0.5 and 2"),
            "{}",
            error.message
        );
        assert!(validate_speech_speed(ModelVariant::FishAudioS2Pro, Some(0.9)).is_err());
    }

    #[test]
    fn qwen_auto_budget_is_text_sized_and_bounded() {
        let short =
            resolve_tts_output_frames(ModelVariant::Qwen3Tts12Hz06BCustomVoice, "hello", None)
                .expect("Qwen frame hint");
        let longer = resolve_tts_output_frames(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            "This sentence is intentionally much longer than hello and needs more speech frames.",
            None,
        )
        .expect("Qwen frame hint");

        assert!(short < ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
        assert!(longer > short);
        assert!(longer <= ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn fish_s2_auto_budget_handles_unspaced_multilingual_text() {
        let short = fish_s2_tts_auto_max_frames_for_text("Hello");
        for text in [
            "你好世界".repeat(30),
            "こんにちは世界".repeat(20),
            "안녕하세요".repeat(24),
            "longunspacedtext".repeat(20),
        ] {
            let budget = fish_s2_tts_auto_max_frames_for_text(&text);
            assert!(budget > short, "{text}");
            assert!(budget <= ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES);
        }
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(10)),
            Some(10)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(usize::MAX)),
            Some(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", Some(0)),
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "你好", None)
        );
    }

    #[test]
    fn chatterbox_controls_are_refused_elsewhere_and_range_checked() {
        let chatterbox = ModelVariant::Chatterbox500M;
        let controls = SpeechSamplingRequest {
            exaggeration: Some(0.9),
            cfg_weight: Some(0.0),
            min_p: Some(0.1),
            ..Default::default()
        };
        assert!(validate_speech_controls(chatterbox, &controls).is_ok());
        assert!(validate_speech_controls(chatterbox, &SpeechSamplingRequest::default()).is_ok());

        let error = validate_speech_controls(ModelVariant::Kokoro82M, &controls).unwrap_err();
        assert!(
            error.message.contains("does not support `exaggeration`"),
            "{}",
            error.message
        );

        for (field, request) in [
            (
                "exaggeration",
                SpeechSamplingRequest {
                    exaggeration: Some(2.5),
                    ..Default::default()
                },
            ),
            (
                "cfg_weight",
                SpeechSamplingRequest {
                    cfg_weight: Some(-0.1),
                    ..Default::default()
                },
            ),
            (
                "min_p",
                SpeechSamplingRequest {
                    min_p: Some(f32::NAN),
                    ..Default::default()
                },
            ),
        ] {
            let error = validate_speech_controls(chatterbox, &request).unwrap_err();
            assert!(error.message.contains(field), "{}", error.message);
        }
    }

    /// Chatterbox refuses a language its edition cannot speak; other
    /// models take `language` as a hint.
    #[test]
    fn chatterbox_language_follows_its_edition() {
        let english = ModelVariant::Chatterbox500M;
        let multilingual = ModelVariant::ChatterboxMultilingual500M;
        for language in [None, Some("en"), Some("English"), Some("auto")] {
            assert!(validate_speech_language(english, language).is_ok());
            assert!(validate_speech_language(multilingual, language).is_ok());
        }
        for language in [Some("fr"), Some("Swahili"), Some("ko"), Some("zh")] {
            assert!(validate_speech_language(multilingual, language).is_ok());
            let error = validate_speech_language(english, language).unwrap_err();
            assert!(error.message.contains("English only"), "{}", error.message);
        }
        for (language, expected) in [("ja", "text frontend"), ("klingon", "does not speak")] {
            let error = validate_speech_language(multilingual, Some(language)).unwrap_err();
            assert!(error.message.contains(expected), "{}", error.message);
        }
        assert!(validate_speech_language(ModelVariant::Kokoro82M, Some("klingon")).is_ok());
    }

    /// Streaming is refused only where the model's capabilities say it
    /// cannot stream.
    #[test]
    fn streaming_follows_the_model_capability() {
        for variant in ModelVariant::all()
            .iter()
            .copied()
            .filter(|variant| variant.is_enabled())
        {
            let Some(capabilities) = variant.speech_capabilities() else {
                continue;
            };
            assert!(validate_speech_streaming(variant, false).is_ok(), "{variant}");
            match validate_speech_streaming(variant, true) {
                Ok(()) => assert!(capabilities.supports_streaming, "{variant}"),
                Err(error) => {
                    assert!(!capabilities.supports_streaming, "{variant}");
                    assert!(error.message.contains("does not support streaming"));
                }
            }
        }
        assert!(validate_speech_streaming(ModelVariant::Chatterbox500M, true).is_ok());
    }

    #[test]
    fn chatterbox_auto_budget_is_text_sized_and_capped() {
        let variant = ModelVariant::Chatterbox500M;
        let short = resolve_tts_output_frames(variant, "Hello there.", None).unwrap();
        let long = resolve_tts_output_frames(variant, &"word ".repeat(1_000), None).unwrap();
        assert!(short < 100, "{short}");
        assert_eq!(long, ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES);
        assert_eq!(resolve_tts_output_frames(variant, "hi", Some(40)), Some(40));
        assert_eq!(
            resolve_tts_output_frames(variant, "hi", Some(usize::MAX)),
            Some(ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES)
        );
    }

    #[test]
    fn explicit_budget_is_clamped_for_every_route() {
        assert_eq!(
            resolve_tts_output_frames(
                ModelVariant::Qwen3Tts12Hz06BCustomVoice,
                "hello",
                Some(usize::MAX),
            ),
            Some(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        );
    }
}
