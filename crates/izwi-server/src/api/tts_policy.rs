use izwi_core::catalog::ModelFamily;
use izwi_core::runtime_models::architectures::chatterbox::{
    chatterbox_auto_max_speech_tokens, ChatterboxEdition,
};
use izwi_core::runtime_models::architectures::fish_drama3::model::auto_frame_budget as fish_drama3_auto_frame_budget;
use izwi_core::runtime_models::architectures::fish_drama3::prompt as fish_drama3_prompt;
use izwi_core::runtime_models::architectures::fish_drama3::reference_cache::MAX_REFERENCES as FISH_DRAMA3_MAX_REFERENCES;
use izwi_core::runtime_models::architectures::fish_s2::contracts::CONTROL_TOKENS as FISH_S2_CONTROL_TOKENS;
use izwi_core::runtime_models::architectures::maya1::{
    maya1_auto_max_new_tokens, prompt::validate_inline_tags,
};
use izwi_core::runtime_models::architectures::lfm3_audio::{
    lfm3_audio_tts_auto_max_frames, prompts::tts_locale as lfm3_audio_tts_locale,
};
use izwi_core::runtime_models::architectures::maya2::resolve_language as maya2_language;
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
/// mode is a reference (Qwen Base, VibeVoice, Fish S2) or a description
/// (Qwen VoiceDesign) must be given one. A model with a seeded voice (Fish
/// Audio Drama 3) speaks without any, so its reference is optional.
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
    if reference {
        validate_speech_references(
            variant,
            &[SpeechReferenceRequest {
                transcript: voice.reference_text,
                saved_voice: voice.saved_voice,
            }],
        )?;
    }
    if voice.voice_description && !capabilities.supports_voice_description {
        return Err(ApiError::bad_request(format!(
            "{variant} does not support voice direction prompts.",
        )));
    }
    if !capabilities.supports_builtin_voices
        && !capabilities.supports_reference_voice
        && !capabilities.supports_voice_description
        && !capabilities.supports_seeded_voice
    {
        return Err(ApiError::bad_request(format!(
            "{variant} does not expose a supported text-to-speech voice mode.",
        )));
    }
    if capabilities.supports_reference_voice
        && !capabilities.supports_builtin_voices
        && !capabilities.supports_voice_description
        && !capabilities.supports_seeded_voice
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

/// One reference voice a request carries, as the routes read it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SpeechReferenceRequest {
    /// Its transcript was given.
    pub(crate) transcript: bool,
    /// It came from a saved voice.
    pub(crate) saved_voice: bool,
}

/// The most reference clips one request of `variant` may carry: the
/// family's prompt limit for Fish Audio Drama 3 (8), one for every other
/// model that clones.
pub(crate) fn max_speech_references(variant: ModelVariant) -> usize {
    match variant.family() {
        ModelFamily::FishDrama3Tts => FISH_DRAMA3_MAX_REFERENCES,
        _ => 1,
    }
}

/// Refuse references the model cannot take: more than
/// [`max_speech_references`], or one without the transcript a model that
/// conditions on its words needs.
pub(crate) fn validate_speech_references(
    variant: ModelVariant,
    references: &[SpeechReferenceRequest],
) -> Result<(), crate::error::ApiError> {
    use crate::error::ApiError;
    let max = max_speech_references(variant);
    if references.len() > max {
        return Err(ApiError::bad_request(format!(
            "{variant} takes at most {max} reference voices; got {}.",
            references.len()
        )));
    }
    let transcript_required = variant
        .speech_capabilities()
        .is_some_and(|capabilities| capabilities.reference_transcript_required);
    if let Some(reference) = references
        .iter()
        .find(|reference| transcript_required && !reference.transcript)
    {
        return Err(ApiError::bad_request(if reference.saved_voice {
            format!(
                "This saved voice has no transcript, which {variant} needs. Use a voice saved \
                 with one, or a model that clones from audio alone."
            )
        } else {
            "Provide both `reference_audio` and `reference_text` together.".to_string()
        }));
    }
    Ok(())
}

/// The tokens only `variant`'s prompt builder writes (Fish S2, Fish Audio
/// Drama 3): the ChatML turn markers, the modality markers and the semantic
/// band. Their tokenizers map each to one special id, so a copy in caller
/// text would rewrite the prompt.
fn speech_control_tokens(variant: ModelVariant) -> &'static [&'static str] {
    match variant.family() {
        ModelFamily::FishS2Tts => &FISH_S2_CONTROL_TOKENS,
        ModelFamily::FishDrama3Tts => &fish_drama3_prompt::STRUCTURAL_TOKENS,
        _ => &[],
    }
}

/// The token of `text` that starts at byte `start`, as far as its closing
/// `|>` (a `<|semantic:` prefix shows its code), bounded for the message.
fn offending_token(text: &str, start: usize) -> &str {
    let rest = &text[start..];
    let bound = rest
        .char_indices()
        .take(32)
        .last()
        .map_or(0, |(offset, c)| offset + c.len_utf8());
    match rest[..bound].find("|>") {
        Some(end) => &rest[..end + 2],
        None => &rest[..bound],
    }
}

/// Refuse caller text (`field`: `input` or `reference_text`) that holds one
/// of the model's control tokens with 400 naming the token, on every speech
/// route, before any model loads. Fish Audio Drama 3's speaker tags must
/// also be `<|speaker:0|>` to `<|speaker:15|>`. Performance tags (`[sighs]`,
/// `<whisper>…</whisper>`, `[language: xx]`) are content and pass through.
pub(crate) fn validate_speech_control_tokens(
    variant: ModelVariant,
    field: &str,
    text: &str,
) -> Result<(), crate::error::ApiError> {
    use crate::error::ApiError;
    let found = speech_control_tokens(variant)
        .iter()
        .filter_map(|token| text.find(token))
        .min();
    if let Some(start) = found {
        return Err(ApiError::bad_request(format!(
            "`{field}` contains `{}`, a control token {variant} reserves for its prompt; remove \
             it. Performance tags such as [sighs] are fine.",
            offending_token(text, start)
        )));
    }
    if variant.family() == ModelFamily::FishDrama3Tts {
        for (start, _) in text.match_indices("<|speaker") {
            let canonical = fish_drama3_prompt::match_speaker_tag(&text[start..])
                .and_then(|(_, digits)| fish_drama3_prompt::speaker_index(digits))
                .is_some();
            if !canonical {
                return Err(ApiError::bad_request(format!(
                    "`{field}` has `{}`; {variant} speaker tags are <|speaker:0|> to \
                     <|speaker:{}|>.",
                    offending_token(text, start),
                    fish_drama3_prompt::MAX_SPEAKER_ID
                )));
            }
        }
    }
    Ok(())
}

/// The `top_k` a speech request samples with: Fish Audio Drama 3 reads
/// `0` as no cap (its defaults: the main token uncapped, the residual heads
/// at 30), since the family itself refuses a cap of 0. Every other model
/// keeps the value as sent.
pub(crate) fn speech_top_k(variant: ModelVariant, top_k: Option<usize>) -> Option<usize> {
    match top_k {
        Some(0) if variant.family() == ModelFamily::FishDrama3Tts => None,
        top_k => top_k,
    }
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

/// Maya 2 Native long-form segments: whole sentences up to about 20 s of
/// speech, upstream's piece length (its frame prior puts 250 frames at
/// 280-380 characters across its languages).
pub(crate) const MAYA2_SEGMENT_MAX_CHARS: usize = 280;

/// Refuse a `language` the model cannot speak before it loads: the English
/// Chatterbox takes English only, the multilingual one its supported codes,
/// Maya 2 Native its 12 codes (or none / `auto`, read from the text's
/// script), LFM3-Audio its 19 locales (none is American English). Other
/// models read `language` as a hint.
pub(crate) fn validate_speech_language(
    variant: ModelVariant,
    language: Option<&str>,
) -> Result<(), crate::error::ApiError> {
    if variant.family() == ModelFamily::Lfm3Audio {
        let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(());
        };
        return lfm3_audio_tts_locale(language)
            .map(|_| ())
            .map_err(|error| match error {
                izwi_core::Error::InvalidInput(message) => {
                    crate::error::ApiError::bad_request(message)
                }
                error => crate::error::core_api_error(error),
            });
    }
    if variant.family() == ModelFamily::Maya2Tts {
        return maya2_language(language, "")
            .map(|_| ())
            .map_err(|error| match error {
                izwi_core::Error::InvalidInput(message) => {
                    crate::error::ApiError::bad_request(message)
                }
                error => crate::error::core_api_error(error),
            });
    }
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

/// Refuse text whose inline `<tag>`s the model does not know (Maya1 would
/// read an unknown one aloud); text for any other model is spoken as written
/// (Maya 2 Native performs its `[tags]` best effort).
pub(crate) fn validate_speech_inline_tags(
    variant: ModelVariant,
    text: &str,
) -> Result<(), crate::error::ApiError> {
    if !variant.refuses_unknown_inline_tags() {
        return Ok(());
    }
    match validate_inline_tags(text) {
        Ok(()) => Ok(()),
        Err(izwi_core::Error::InvalidInput(message)) => {
            Err(crate::error::ApiError::bad_request(message))
        }
        Err(err) => Err(crate::error::core_api_error(err)),
    }
}

/// Refuse a `reference_language` the request cannot use: only Maya 2 Native
/// reads one, and it must name one of its languages (or `auto`).
/// Whether `variant` reads `reference_language`: Maya 2 Native, which
/// clones a voice into another language.
pub(crate) fn takes_reference_language(variant: ModelVariant) -> bool {
    variant.family() == ModelFamily::Maya2Tts
}

pub(crate) fn validate_speech_reference_language(
    variant: ModelVariant,
    reference_language: Option<&str>,
) -> Result<(), crate::error::ApiError> {
    let Some(language) = reference_language.filter(|value| !value.trim().is_empty()) else {
        return Ok(());
    };
    if !takes_reference_language(variant) {
        return Err(crate::error::ApiError::bad_request(format!(
            "{variant} does not take `reference_language`; only Maya 2 Native clones across \
             languages."
        )));
    }
    validate_speech_language(variant, Some(language))
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
    /// Sampling seed (Qwen3-TTS, LFM2.5-Audio, Fish S2, Fish Audio Drama 3,
    /// Voxtral noise, Kokoro noise). Fish Audio Drama 3's voice without a
    /// reference comes from it, and chunk i of long text samples from
    /// seed + i.
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
    /// Speak numbers, currency, dates, times and units as words. Unset keeps
    /// the model's default: on for Kokoro (English) and Maya 2 Native (in the
    /// request's language), off for models that read them themselves.
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
        // Per chunk: S2's speaking estimate times 2.5, at most 1,024 frames.
        Some(0) | None if variant.family() == ModelFamily::FishDrama3Tts => {
            Some(fish_drama3_auto_frame_budget(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::ChatterboxTts => {
            Some(chatterbox_auto_max_speech_tokens(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::Maya1Tts => {
            Some(maya1_auto_max_new_tokens(text).min(model_max_frames))
        }
        Some(0) | None if variant.family() == ModelFamily::Lfm3Audio => {
            Some(lfm3_audio_tts_auto_max_frames(text).min(model_max_frames))
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
        // The automatic budget is whole 7-token frames, 2,044 at most.
        ModelFamily::Maya1Tts => ModelVariant::MAYA1_MAX_OUTPUT_FRAMES / 7 * 7,
        // Upstream speaks long text in pieces of about 20 s; each piece's
        // budget comes from its own frame prior.
        ModelFamily::Maya2Tts => return text.trim().chars().count() > MAYA2_SEGMENT_MAX_CHARS,
        ModelFamily::Lfm3Audio => ModelVariant::LFM3_AUDIO_MAX_OUTPUT_FRAMES,
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

    /// Fish Audio Drama 3 speaks with or without a reference (its seed sets
    /// the voice without one), and a reference needs its transcript, saved
    /// or inline. Fish S2 still requires a reference.
    #[test]
    fn fish_drama3_takes_an_optional_reference_with_its_transcript() {
        let drama3 = ModelVariant::FishAudioDrama3;
        let voice = |reference_audio, reference_text, saved_voice| SpeechVoiceRequest {
            reference_audio,
            reference_text,
            saved_voice,
            ..Default::default()
        };
        validate_speech_voice(drama3, voice(false, false, false)).expect("a seeded voice");
        validate_speech_voice(drama3, voice(true, true, false)).expect("a clone");
        validate_speech_voice(drama3, voice(true, true, true)).expect("a saved voice");
        let err = validate_speech_voice(drama3, voice(true, false, false)).unwrap_err();
        assert!(err.message.contains("Provide both"), "{}", err.message);
        let err = validate_speech_voice(drama3, voice(true, false, true)).unwrap_err();
        assert!(err.message.contains("saved voice has no transcript"));
        let err = validate_speech_voice(drama3, voice(false, true, false)).unwrap_err();
        assert!(err.message.contains("without `reference_audio`"));
        for refused in [
            SpeechVoiceRequest {
                speaker: true,
                ..Default::default()
            },
            SpeechVoiceRequest {
                voice_description: true,
                ..Default::default()
            },
        ] {
            assert!(validate_speech_voice(drama3, refused).is_err());
        }
        // S2's contract is unchanged: it must be given a reference.
        let err = validate_speech_voice(ModelVariant::FishAudioS2Pro, voice(false, false, false))
            .unwrap_err();
        assert!(err
            .message
            .contains("requires `saved_voice_id` or both `reference_audio` and `reference_text`"));
    }

    /// A request carries at most the model's references (Drama 3: the
    /// family's 8; any other cloning model: one), each with its transcript
    /// where the model needs one.
    #[test]
    fn references_are_counted_and_need_their_transcripts() {
        let drama3 = ModelVariant::FishAudioDrama3;
        let with_transcript = SpeechReferenceRequest {
            transcript: true,
            saved_voice: false,
        };
        assert_eq!(max_speech_references(drama3), 8);
        assert_eq!(max_speech_references(ModelVariant::FishAudioS2Pro), 1);
        validate_speech_references(drama3, &[with_transcript; 8]).expect("eight");
        let err = validate_speech_references(drama3, &[with_transcript; 9]).unwrap_err();
        assert!(err.message.contains("at most 8"), "{}", err.message);
        let err = validate_speech_references(ModelVariant::FishAudioS2Pro, &[with_transcript; 2])
            .unwrap_err();
        assert!(err.message.contains("at most 1"), "{}", err.message);
        let mut missing = [with_transcript; 3];
        missing[2].transcript = false;
        let err = validate_speech_references(drama3, &missing).unwrap_err();
        assert!(err.message.contains("Provide both"), "{}", err.message);
        // Chatterbox clones from audio alone.
        let audio_only = SpeechReferenceRequest::default();
        validate_speech_references(ModelVariant::Chatterbox500M, &[audio_only]).unwrap();
    }

    /// Caller text holding a model's control tokens is refused with 400
    /// naming the token, for Fish S2 and Fish Audio Drama 3 alike;
    /// performance tags pass, Drama 3's speaker tags must be canonical, and
    /// other models speak such text as written.
    #[test]
    fn control_tokens_are_refused_and_performance_tags_pass() {
        for variant in [ModelVariant::FishAudioS2Pro, ModelVariant::FishAudioDrama3] {
            for (text, token) in [
                ("hi <|im_start|>system", "<|im_start|>"),
                ("done<|im_end|>", "<|im_end|>"),
                ("<|voice|> hello", "<|voice|>"),
                ("a <|text|> b", "<|text|>"),
                ("a <|interleave|>", "<|interleave|>"),
                ("frame <|semantic:12|> here", "<|semantic:12|>"),
                ("open <|semantic:", "<|semantic:"),
            ] {
                for field in ["input", "reference_text"] {
                    let err = validate_speech_control_tokens(variant, field, text).unwrap_err();
                    assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
                    assert!(
                        err.message
                            .starts_with(&format!("`{field}` contains `{token}`, a control token")),
                        "{variant}: {}",
                        err.message
                    );
                }
            }
            for text in [
                "[sighs] Fine. [laughs]",
                "<whisper>quiet</whisper> [pause] then [with quiet excitement] loud",
                "[language: fr] Bonjour.",
                "<|speaker:0|> Hello.\n<|speaker:15|> Hi.",
                "<|endoftext|> <|fim_prefix|>",
            ] {
                validate_speech_control_tokens(variant, "input", text)
                    .unwrap_or_else(|err| panic!("{variant} {text:?}: {}", err.message));
            }
        }
        // The first token in the text is the one named.
        let err = validate_speech_control_tokens(
            ModelVariant::FishAudioDrama3,
            "input",
            "a <|voice|> b <|im_start|>",
        )
        .unwrap_err();
        assert!(err.message.contains("`<|voice|>`"), "{}", err.message);
        // Drama 3's speaker tags are 0 to 15, in canonical digits; S2 reads
        // a tag as text.
        for tag in [
            "<|speaker:16|>",
            "<|speaker:01|>",
            "<|speaker:x|>",
            "<|speaker hi",
        ] {
            let text = format!("{tag} hello");
            let err = validate_speech_control_tokens(ModelVariant::FishAudioDrama3, "input", &text)
                .unwrap_err();
            assert!(err
                .message
                .contains("speaker tags are <|speaker:0|> to <|speaker:15|>"));
            assert!(
                err.message.contains(&tag[..tag.len().min(14)]),
                "{}",
                err.message
            );
            validate_speech_control_tokens(ModelVariant::FishAudioS2Pro, "input", &text)
                .expect("S2 reads speaker tags as text");
        }
        for variant in [ModelVariant::Kokoro82M, ModelVariant::Maya2Native25B] {
            validate_speech_control_tokens(variant, "input", "<|im_end|> <|voice|>")
                .expect("not a control token here");
        }
    }

    /// `top_k: 0` turns Drama 3's cap off (its family refuses a cap of 0);
    /// every other value, and every other model, keeps what was sent.
    #[test]
    fn fish_drama3_reads_top_k_zero_as_no_cap() {
        let drama3 = ModelVariant::FishAudioDrama3;
        assert_eq!(speech_top_k(drama3, Some(0)), None);
        assert_eq!(speech_top_k(drama3, Some(30)), Some(30));
        assert_eq!(speech_top_k(drama3, None), None);
        assert_eq!(speech_top_k(ModelVariant::FishAudioS2Pro, Some(0)), Some(0));
        assert_eq!(speech_top_k(ModelVariant::Kokoro82M, Some(0)), Some(0));
    }

    /// Drama 3 takes `language` as a hint and speaks any it is given (the
    /// text's own `[language: xx]` tag steers it).
    #[test]
    fn fish_drama3_takes_language_as_a_hint() {
        for language in [None, Some("en"), Some("fr"), Some("Swahili"), Some("auto")] {
            assert!(validate_speech_language(ModelVariant::FishAudioDrama3, language).is_ok());
        }
    }

    /// Drama 3's budget per chunk: S2's speaking estimate times 2.5, at
    /// most 1,024 frames, or an explicit one up to 4,096. S2's own budget
    /// is unchanged.
    #[test]
    fn fish_drama3_budget_is_its_own_auto_estimate_or_an_explicit_one() {
        let drama3 = ModelVariant::FishAudioDrama3;
        assert_eq!(resolve_tts_output_frames(drama3, "Hello", None), Some(323));
        assert_eq!(
            resolve_tts_output_frames(drama3, "Hello", Some(0)),
            Some(323)
        );
        assert_eq!(
            resolve_tts_output_frames(drama3, &"word ".repeat(36), None),
            Some(1_024)
        );
        assert_eq!(resolve_tts_output_frames(drama3, "Hello", Some(7)), Some(7));
        assert_eq!(
            resolve_tts_output_frames(drama3, "Hello", Some(2_000)),
            Some(2_000)
        );
        assert_eq!(
            resolve_tts_output_frames(drama3, "Hello", Some(9_999)),
            Some(4_096)
        );
        // The family's own estimate.
        for text in [
            "Hello",
            "你好世界".repeat(30).as_str(),
            &"word ".repeat(200),
        ] {
            assert_eq!(
                resolve_tts_output_frames(drama3, text, None),
                Some(fish_drama3_auto_frame_budget(text))
            );
        }
        // S2: 6 s at 21.5 Hz for a short text, as before.
        assert_eq!(
            resolve_tts_output_frames(ModelVariant::FishAudioS2Pro, "Hello", None),
            Some(129)
        );
        assert!(!tts_text_exceeds_one_request(
            drama3,
            &"word ".repeat(1_000)
        ));
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
    fn maya1_budget_is_text_sized_in_whole_frames_and_capped() {
        let variant = ModelVariant::Maya13B;
        let short = resolve_tts_output_frames(variant, "Hello there.", None).unwrap();
        assert!(short < 400 && short.is_multiple_of(7), "{short}");
        let tagged =
            resolve_tts_output_frames(variant, "Hello there. <laugh_harder>", None).unwrap();
        assert!(tagged > short + 100, "{tagged} vs {short}");
        let long = resolve_tts_output_frames(variant, &"word ".repeat(1_000), None).unwrap();
        assert_eq!(long, 2_044);
        assert!(tts_text_exceeds_one_request(
            variant,
            &"word ".repeat(1_000)
        ));
        assert!(!tts_text_exceeds_one_request(variant, "Hello there."));
        assert_eq!(
            resolve_tts_output_frames(variant, "hi", Some(usize::MAX)),
            Some(ModelVariant::MAYA1_MAX_OUTPUT_FRAMES)
        );
    }

    /// Maya1 refuses a tag it does not know (it would read it aloud); any
    /// other model speaks angle brackets as written.
    #[test]
    fn inline_tags_are_checked_only_for_models_that_have_them() {
        let maya1 = ModelVariant::Maya13B;
        assert!(validate_speech_inline_tags(maya1, "Ha! <laugh> fine. <sigh>").is_ok());
        let error = validate_speech_inline_tags(maya1, "Well <happy> then").unwrap_err();
        assert!(format!("{error:?}").contains("<happy>"), "{error:?}");
        assert!(validate_speech_inline_tags(ModelVariant::Kokoro82M, "a <happy> b").is_ok());
    }

    /// Maya1's voices are named descriptions: a name, a description, or
    /// both are accepted; a reference is not.
    #[test]
    fn maya1_takes_voice_names_and_descriptions_not_references() {
        let maya1 = ModelVariant::Maya13B;
        let request = |speaker, voice_description, reference_audio| SpeechVoiceRequest {
            speaker,
            voice_description,
            reference_audio,
            ..SpeechVoiceRequest::default()
        };
        assert!(validate_speech_voice(maya1, request(false, false, false)).is_ok());
        assert!(validate_speech_voice(maya1, request(true, false, false)).is_ok());
        assert!(validate_speech_voice(maya1, request(false, true, false)).is_ok());
        assert!(validate_speech_voice(maya1, request(true, true, false)).is_ok());
        assert!(validate_speech_voice(maya1, request(false, false, true)).is_err());
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
