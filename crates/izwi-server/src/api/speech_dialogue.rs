//! Speaker dialogue on the speech routes, for a model whose speech
//! capabilities report `supports_speaker_dialogue` (Fish Audio Drama 3):
//! the `references` and `script` request fields, their checks, saved voices
//! per speaker, and the references the runtime is given.
//!
//! - `references: [{speaker, audio | saved_voice_id, text}]`: at most the
//!   model's `max_references` clips, each voicing one speaker `0` to
//!   `max_speakers - 1` (default 0). Several may voice one speaker, as
//!   upstream's prompt allows (two clips of one voice share a tag). Every
//!   chunk of a long text holds them all, as upstream's chunk session does.
//!   They replace `reference_audio` / `reference_text` / `saved_voice_id`,
//!   which mean one reference for speaker 0; a request gives one form.
//! - `script: [{speaker, text}]` is rendered as `<|speaker:N|> text` lines
//!   joined by `"\n"` and becomes the request's text: a record stores the
//!   rendered text. It replaces `input` / `text`.
//! - A speaker the text names without a reference speaks in a voice the
//!   seed sets, beside referenced ones, as upstream allows: the prompt
//!   carries the references and the text tags every speaker. A dialogue
//!   model without a seeded voice would need a reference per speaker.
//! - Inline clips total at most [`MAX_INLINE_REFERENCE_BYTES`] of base64
//!   per request; saved voices do not count.

use izwi_core::runtime_models::architectures::fish_drama3::prompt as fish_drama3_prompt;
use izwi_core::{ModelVariant, SpeechSpeakerReference};
use serde::{Deserialize, Serialize};

use crate::api::tts_policy::{validate_speech_control_tokens, SpeechReferenceRequest};
use crate::error::ApiError;
use crate::state::AppState;

/// The most base64 audio one request's inline references may carry (16 MiB,
/// about 140 s of 16-bit 44.1 kHz WAV): below every speech route's body
/// limit (25 MiB on `/v1/audio/speech` by default, 64 MiB on the
/// first-party routes), so a request over it hears where to put its voices
/// instead of a bare 413.
pub(crate) const MAX_INLINE_REFERENCE_BYTES: usize = 16 * 1024 * 1024;

/// One reference voice of a dialogue request: a clip (`audio`, base64, with
/// its transcript `text`) or a saved voice, and the speaker it voices.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeechDialogueReference {
    /// The speaker this reference voices, 0 to the model's
    /// `max_speakers - 1`; unset, the speaker of a `<|speaker:N|>` tag that
    /// leads `text`, else 0. A leading tag must name this speaker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<usize>,
    /// The clip, base64 audio (at most 60 s for Fish Audio Drama 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<String>,
    /// What the clip says; required with `audio`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A saved voice instead of `audio` and `text`: its clip and
    /// transcript (it must have one).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saved_voice_id: Option<String>,
}

/// One line of a dialogue script: a speaker and what they say.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
pub struct SpeechScriptLine {
    /// The speaker, 0 to the model's `max_speakers - 1`.
    pub speaker: usize,
    /// The line, without a speaker tag; performance tags such as `[sighs]`
    /// pass through.
    pub text: String,
}

/// The dialogue fields of a speech request, as the routes read them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SpeechDialogue<'a> {
    /// `reference_audio`, `reference_text` or `saved_voice_id` was given.
    pub single_reference: bool,
    pub references: &'a [SpeechDialogueReference],
    pub script: Option<&'a [SpeechScriptLine]>,
}

impl SpeechDialogue<'_> {
    fn is_empty(&self) -> bool {
        self.references.is_empty() && self.script.is_none()
    }
}

fn bad_request(error: izwi_core::Error) -> ApiError {
    match error {
        izwi_core::Error::InvalidInput(message) => ApiError::bad_request(message),
        other => ApiError::bad_request(other.to_string()),
    }
}

/// Trims every reference field, dropping empty ones, as the routes trim the
/// single-reference fields.
pub(crate) fn normalize_references(references: &mut [SpeechDialogueReference]) {
    let trimmed = |value: &mut Option<String>| {
        *value = value
            .take()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());
    };
    for reference in references {
        trimmed(&mut reference.audio);
        trimmed(&mut reference.text);
        trimmed(&mut reference.saved_voice_id);
    }
}

/// The text a request speaks: its `script` rendered as tagged lines, or its
/// `input` (`text` on the first-party routes, named `input_field`). Giving
/// both is a 400; so is a script line that is empty, carries its own
/// speaker tag or names a speaker past 15.
pub(crate) fn script_text(
    input: Option<&str>,
    script: Option<&[SpeechScriptLine]>,
    input_field: &str,
) -> Result<Option<String>, ApiError> {
    let Some(script) = script else {
        return Ok(None);
    };
    if input.is_some_and(|input| !input.trim().is_empty()) {
        return Err(ApiError::bad_request(format!(
            "Use either `{input_field}` or `script`, not both."
        )));
    }
    let lines = script
        .iter()
        .map(|line| (line.speaker, line.text.as_str()))
        .collect::<Vec<_>>();
    fish_drama3_prompt::render_script(&lines)
        .map(Some)
        .map_err(bad_request)
}

/// Refuse a dialogue request's shape before any saved voice is looked up or
/// model loaded: dialogue fields for a model that speaks one voice, both
/// reference forms, more references than the model takes, a speaker past
/// its `max_speakers`, a reference that is not exactly one of a clip or a
/// saved voice, a saved voice given a transcript, inline clips past
/// [`MAX_INLINE_REFERENCE_BYTES`], and control tokens in a script line.
pub(crate) fn check_dialogue_request(
    variant: ModelVariant,
    dialogue: SpeechDialogue<'_>,
) -> Result<(), ApiError> {
    if dialogue.is_empty() {
        return Ok(());
    }
    let capabilities = variant
        .speech_capabilities()
        .filter(|capabilities| capabilities.supports_speaker_dialogue)
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "{variant} speaks one voice: `references` and `script` are for models that \
                 voice several speakers. Use `reference_audio` and `reference_text`, or \
                 `saved_voice_id`."
            ))
        })?;
    if dialogue.single_reference && !dialogue.references.is_empty() {
        return Err(ApiError::bad_request(
            "Use either `references` or `reference_audio`/`reference_text`/`saved_voice_id`, \
             not both.",
        ));
    }
    let max_references = capabilities.max_references;
    if dialogue.references.len() > max_references {
        return Err(ApiError::bad_request(format!(
            "{variant} takes at most {max_references} references; got {}.",
            dialogue.references.len()
        )));
    }
    let last_speaker = capabilities.max_speakers.saturating_sub(1);
    let mut inline_bytes = 0usize;
    for (index, reference) in dialogue.references.iter().enumerate() {
        if let Some(speaker) = reference.speaker.filter(|&speaker| speaker > last_speaker) {
            return Err(ApiError::bad_request(format!(
                "`references[{index}].speaker` is {speaker}; {variant} speakers are 0 to \
                 {last_speaker}."
            )));
        }
        match (&reference.audio, &reference.saved_voice_id) {
            (Some(_), Some(_)) => {
                return Err(ApiError::bad_request(format!(
                    "`references[{index}]` gives both `audio` and `saved_voice_id`; give one."
                )))
            }
            (None, None) => {
                return Err(ApiError::bad_request(format!(
                    "`references[{index}]` needs `audio` (with its `text`) or `saved_voice_id`."
                )))
            }
            (None, Some(_)) if reference.text.is_some() => {
                return Err(ApiError::bad_request(format!(
                    "`references[{index}]` is a saved voice, which brings its own transcript; \
                     drop `text`."
                )))
            }
            (Some(audio), None) => inline_bytes = inline_bytes.saturating_add(audio.len()),
            (None, Some(_)) => {}
        }
    }
    if inline_bytes > MAX_INLINE_REFERENCE_BYTES {
        return Err(ApiError::bad_request(format!(
            "The inline reference audio totals {:.1} MiB; a request carries at most {} MiB. \
             Save the voices (POST /v1/voices) and give each reference its `saved_voice_id`.",
            inline_bytes as f64 / (1024.0 * 1024.0),
            MAX_INLINE_REFERENCE_BYTES / (1024 * 1024)
        )));
    }
    for (index, line) in dialogue.script.unwrap_or_default().iter().enumerate() {
        if line.speaker > last_speaker {
            return Err(ApiError::bad_request(format!(
                "`script[{index}].speaker` is {}; {variant} speakers are 0 to {last_speaker}.",
                line.speaker
            )));
        }
        validate_speech_control_tokens(variant, &format!("script[{index}].text"), &line.text)?;
    }
    Ok(())
}

/// Replaces each saved voice in `references` with its clip and transcript;
/// the id stays, for the record.
pub(crate) async fn resolve_saved_voices(
    state: &AppState,
    references: &mut [SpeechDialogueReference],
) -> Result<(), ApiError> {
    for reference in references {
        let Some(saved_voice_id) = reference.saved_voice_id.as_deref() else {
            continue;
        };
        let saved_voice =
            crate::api::saved_voices::resolve_saved_voice_reference(state, saved_voice_id).await?;
        reference.saved_voice_id = Some(saved_voice.voice_id);
        reference.audio = Some(saved_voice.reference_audio_base64);
        reference.text = saved_voice.reference_text;
    }
    Ok(())
}

/// The speaker of a checked reference: its `speaker`, else that of a tag
/// leading its transcript, else 0.
pub(crate) fn reference_speaker(reference: &SpeechDialogueReference) -> usize {
    reference.speaker.unwrap_or_else(|| {
        reference
            .text
            .as_deref()
            .map(fish_drama3_prompt::lstrip_white_space)
            .and_then(fish_drama3_prompt::match_speaker_tag)
            .and_then(|(_, digits)| fish_drama3_prompt::speaker_index(digits))
            .unwrap_or(0)
    })
}

/// The speakers `text` voices: each `<|speaker:N|>` tag's, and 0 when text
/// comes before the first tag (the prompt tags it speaker 0).
fn text_speakers(text: &str) -> Vec<usize> {
    let mut speakers = Vec::new();
    let first_tag = fish_drama3_prompt::speaker_tags(text)
        .next()
        .map_or(text.len(), |(start, _, _)| start);
    if !fish_drama3_prompt::strip_white_space(&text[..first_tag]).is_empty() {
        speakers.push(0);
    }
    for (_, _, digits) in fish_drama3_prompt::speaker_tags(text) {
        if let Some(speaker) = fish_drama3_prompt::speaker_index(digits) {
            if !speakers.contains(&speaker) {
                speakers.push(speaker);
            }
        }
    }
    speakers
}

/// Refuse a speaker of `text` without a reference when the model has no
/// seeded voice to give it.
fn check_unreferenced_speakers(
    variant: ModelVariant,
    seeded_voice: bool,
    referenced: &[usize],
    text: &str,
) -> Result<(), ApiError> {
    if seeded_voice || referenced.is_empty() {
        return Ok(());
    }
    match text_speakers(text)
        .into_iter()
        .find(|speaker| !referenced.contains(speaker))
    {
        Some(speaker) => Err(ApiError::bad_request(format!(
            "Speaker {speaker} has no reference, and {variant} needs one for every speaker."
        ))),
        None => Ok(()),
    }
}

/// Refuse resolved dialogue references the model cannot speak from: one
/// without its transcript (a saved voice saved without one included),
/// control tokens in a transcript, a transcript led by another speaker's
/// tag, and, for a model without a seeded voice, a speaker of `text`
/// without a reference.
pub(crate) fn check_dialogue_references(
    variant: ModelVariant,
    references: &[SpeechDialogueReference],
    text: &str,
) -> Result<(), ApiError> {
    if references.is_empty() {
        return Ok(());
    }
    crate::api::tts_policy::validate_speech_references(
        variant,
        &references
            .iter()
            .enumerate()
            .map(|(index, reference)| SpeechReferenceRequest {
                transcript: reference.text.is_some(),
                saved_voice: reference.saved_voice_id.is_some(),
                index: Some(index),
            })
            .collect::<Vec<_>>(),
    )?;
    for (index, reference) in references.iter().enumerate() {
        let Some(transcript) = reference.text.as_deref() else {
            continue;
        };
        validate_speech_control_tokens(variant, &format!("references[{index}].text"), transcript)?;
        if let Some(speaker) = reference.speaker {
            fish_drama3_prompt::speaker_reference_transcript(speaker, transcript).map_err(
                |_| {
                    ApiError::bad_request(format!(
                        "`references[{index}].text` starts with another speaker's tag than \
                         its `speaker` {speaker}; drop the tag or match it."
                    ))
                },
            )?;
        }
    }
    let seeded_voice = variant
        .speech_capabilities()
        .is_some_and(|capabilities| capabilities.supports_seeded_voice);
    let referenced = references.iter().map(reference_speaker).collect::<Vec<_>>();
    check_unreferenced_speakers(variant, seeded_voice, &referenced, text)
}

/// The references the runtime is given, each with its speaker; checked
/// and resolved first.
pub(crate) fn speaker_references(
    references: &[SpeechDialogueReference],
) -> Vec<SpeechSpeakerReference> {
    references
        .iter()
        .map(|reference| SpeechSpeakerReference {
            speaker: reference_speaker(reference),
            audio: reference.audio.clone().unwrap_or_default(),
            text: reference.text.clone().unwrap_or_default(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAMA3: ModelVariant = ModelVariant::FishAudioDrama3;

    fn inline(speaker: usize, text: &str) -> SpeechDialogueReference {
        SpeechDialogueReference {
            speaker: Some(speaker),
            audio: Some("UklGRg==".into()),
            text: Some(text.into()),
            saved_voice_id: None,
        }
    }

    fn saved(speaker: usize, id: &str) -> SpeechDialogueReference {
        SpeechDialogueReference {
            speaker: Some(speaker),
            audio: None,
            text: None,
            saved_voice_id: Some(id.into()),
        }
    }

    fn line(speaker: usize, text: &str) -> SpeechScriptLine {
        SpeechScriptLine {
            speaker,
            text: text.into(),
        }
    }

    fn dialogue<'a>(
        references: &'a [SpeechDialogueReference],
        script: Option<&'a [SpeechScriptLine]>,
    ) -> SpeechDialogue<'a> {
        SpeechDialogue {
            single_reference: false,
            references,
            script,
        }
    }

    fn refused(result: Result<(), ApiError>) -> String {
        let error = result.expect_err("a 400");
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        error.message
    }

    /// The catalog's limits are the family's: eight references, speakers
    /// 0 to 15.
    #[test]
    fn the_catalog_limits_are_the_family_prompt_limits() {
        let capabilities = DRAMA3.speech_capabilities().unwrap();
        assert!(capabilities.supports_speaker_dialogue);
        assert_eq!(
            capabilities.max_speakers,
            fish_drama3_prompt::MAX_SPEAKER_ID + 1
        );
        assert_eq!(
            capabilities.max_references,
            izwi_core::runtime_models::architectures::fish_drama3::reference_cache::MAX_REFERENCES
        );
    }

    #[test]
    fn a_script_renders_as_tagged_lines_and_excludes_input() {
        let script = [line(0, " Ready? "), line(1, "Always. [laughs]")];
        assert_eq!(
            script_text(None, Some(&script), "input")
                .unwrap()
                .as_deref(),
            Some("<|speaker:0|> Ready?\n<|speaker:1|> Always. [laughs]")
        );
        assert_eq!(script_text(Some("Hi."), None, "input").unwrap(), None);
        // An empty input is no input.
        assert!(script_text(Some("  "), Some(&script), "input").is_ok());
        let message = script_text(Some("Hi."), Some(&script), "text")
            .unwrap_err()
            .message;
        assert!(message.contains("either `text` or `script`"), "{message}");
        for (bad, expected) in [
            (vec![], "no lines"),
            (vec![line(0, "  ")], "must not be empty"),
            (vec![line(0, "<|speaker:1|> Hi.")], "speaker field"),
            (vec![line(16, "Hi.")], "speaker 16 is not 0 to 15"),
        ] {
            let message = script_text(None, Some(&bad), "input").unwrap_err().message;
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn dialogue_fields_are_for_dialogue_models_only() {
        let references = [inline(0, "Hi.")];
        let script = [line(0, "Hi.")];
        for variant in [
            ModelVariant::FishAudioS2Pro,
            ModelVariant::Qwen3Tts12Hz06BBase,
            ModelVariant::Kokoro82M,
        ] {
            let message = refused(check_dialogue_request(variant, dialogue(&references, None)));
            assert!(message.contains("speaks one voice"), "{variant}: {message}");
            let message = refused(check_dialogue_request(
                variant,
                dialogue(&[], Some(&script)),
            ));
            assert!(message.contains("speaks one voice"), "{variant}: {message}");
            // Without them nothing changes.
            check_dialogue_request(variant, dialogue(&[], None)).unwrap();
        }
        check_dialogue_request(DRAMA3, dialogue(&references, Some(&script))).unwrap();
    }

    #[test]
    fn references_replace_the_single_reference_fields() {
        let references = [inline(1, "Hi.")];
        let mut both = dialogue(&references, None);
        both.single_reference = true;
        let message = refused(check_dialogue_request(DRAMA3, both));
        assert!(message.contains("not both"), "{message}");
        // The single fields alone are the 4a contract's.
        let mut single = dialogue(&[], None);
        single.single_reference = true;
        check_dialogue_request(DRAMA3, single).unwrap();
    }

    #[test]
    fn references_are_counted_and_their_speakers_bounded() {
        let eight = (0..8)
            .map(|speaker| inline(speaker, "Hi."))
            .collect::<Vec<_>>();
        check_dialogue_request(DRAMA3, dialogue(&eight, None)).unwrap();
        let nine = (0..9)
            .map(|speaker| inline(speaker % 4, "Hi."))
            .collect::<Vec<_>>();
        let message = refused(check_dialogue_request(DRAMA3, dialogue(&nine, None)));
        assert!(message.contains("at most 8 references; got 9"), "{message}");
        // Several clips may voice one speaker, as upstream's prompt allows.
        let same = [inline(2, "Hi."), inline(2, "Bye.")];
        check_dialogue_request(DRAMA3, dialogue(&same, None)).unwrap();
        let message = refused(check_dialogue_request(
            DRAMA3,
            dialogue(&[inline(0, "Hi."), inline(16, "Hi.")], None),
        ));
        assert!(
            message.contains("`references[1].speaker` is 16") && message.contains("0 to 15"),
            "{message}"
        );
        let message = refused(check_dialogue_request(
            DRAMA3,
            dialogue(
                &[],
                Some(&[line(3, "Hi."), line(15, "Hi."), line(16, "Hi.")]),
            ),
        ));
        assert!(message.contains("`script[2].speaker` is 16"), "{message}");
    }

    #[test]
    fn each_reference_is_a_clip_or_a_saved_voice() {
        let mut both = inline(0, "Hi.");
        both.saved_voice_id = Some("voice".into());
        let mut neither = inline(0, "Hi.");
        neither.audio = None;
        let mut saved_with_text = saved(1, "voice");
        saved_with_text.text = Some("Hi.".into());
        for (reference, expected) in [
            (both, "both `audio` and `saved_voice_id`"),
            (neither, "needs `audio`"),
            (saved_with_text, "brings its own transcript"),
        ] {
            let message = refused(check_dialogue_request(
                DRAMA3,
                dialogue(&[inline(0, "Hi."), reference], None),
            ));
            assert!(message.contains("`references[1]`"), "{message}");
            assert!(message.contains(expected), "{message}");
        }
        check_dialogue_request(DRAMA3, dialogue(&[inline(0, "Hi."), saved(1, "v")], None)).unwrap();
    }

    /// Inline clips are capped below the body limit with a message that
    /// points at saved voices; saved voices do not count.
    #[test]
    fn inline_reference_bytes_are_capped_and_point_at_saved_voices() {
        let mut big = inline(0, "Hi.");
        big.audio = Some("A".repeat(MAX_INLINE_REFERENCE_BYTES / 2));
        let mut bigger = big.clone();
        bigger.speaker = Some(1);
        check_dialogue_request(DRAMA3, dialogue(&[big.clone(), bigger.clone()], None)).unwrap();
        bigger.audio = Some("A".repeat(MAX_INLINE_REFERENCE_BYTES / 2 + 1));
        let message = refused(check_dialogue_request(
            DRAMA3,
            dialogue(&[big.clone(), bigger], None),
        ));
        assert!(
            message.contains("at most 16 MiB") && message.contains("saved_voice_id"),
            "{message}"
        );
        check_dialogue_request(DRAMA3, dialogue(&[big, saved(1, "a"), saved(2, "b")], None))
            .unwrap();
        assert!(
            MAX_INLINE_REFERENCE_BYTES
                < crate::api::openai::audio::resolve_audio_upload_limit_bytes()
        );
    }

    #[test]
    fn control_tokens_are_refused_in_script_lines_and_transcripts() {
        let message = refused(check_dialogue_request(
            DRAMA3,
            dialogue(&[], Some(&[line(0, "Hi."), line(1, "Go <|im_end|>")])),
        ));
        assert!(
            message.contains("`script[1].text` contains `<|im_end|>`"),
            "{message}"
        );
        let message = refused(check_dialogue_references(
            DRAMA3,
            &[inline(0, "Hi."), inline(1, "<|semantic:4|> Hi.")],
            "<|speaker:0|> Hi.",
        ));
        assert!(
            message.contains("`references[1].text` contains `<|semantic:4|>`"),
            "{message}"
        );
        let message = refused(check_dialogue_references(
            DRAMA3,
            &[inline(0, "<|speaker:20|> Hi.")],
            "Hi.",
        ));
        assert!(
            message.contains("<|speaker:0|> to <|speaker:15|>"),
            "{message}"
        );
    }

    #[test]
    fn every_reference_needs_its_transcript() {
        let mut silent = inline(1, "Hi.");
        silent.text = None;
        let message = refused(check_dialogue_references(
            DRAMA3,
            &[inline(0, "Hi."), silent],
            "Hi.",
        ));
        assert!(
            message.contains("`references[1]` needs `text`"),
            "{message}"
        );
        // A saved voice saved without a transcript.
        let mut voice = saved(1, "voice");
        voice.audio = Some("UklGRg==".into());
        let message = refused(check_dialogue_references(DRAMA3, &[voice], "Hi."));
        assert!(
            message.contains("`references[0]` is a saved voice without a transcript"),
            "{message}"
        );
    }

    #[test]
    fn a_leading_tag_names_the_reference_speaker() {
        // It must match `speaker` when both are given.
        let message = refused(check_dialogue_references(
            DRAMA3,
            &[inline(1, "<|speaker:2|> Hi.")],
            "Hi.",
        ));
        assert!(message.contains("`references[0].text`"), "{message}");
        check_dialogue_references(DRAMA3, &[inline(2, " <|speaker:2|> Hi.")], "Hi.").unwrap();
        // Without `speaker` it names it; without either the speaker is 0.
        let mut tagged = inline(0, "<|speaker:5|> Hi.");
        tagged.speaker = None;
        let mut plain = inline(0, "Hi.");
        plain.speaker = None;
        let references = speaker_references(&[tagged, plain, inline(3, "Yo.")]);
        assert_eq!(
            references
                .iter()
                .map(|reference| reference.speaker)
                .collect::<Vec<_>>(),
            [5, 0, 3]
        );
        assert_eq!(references[2].text, "Yo.");
        assert_eq!(references[2].audio, "UklGRg==");
    }

    /// Mixing referenced and seeded speakers is allowed for a model with a
    /// seeded voice (Drama 3, as upstream); a dialogue model without one
    /// would need a reference for every speaker its text names.
    #[test]
    fn unreferenced_speakers_need_a_seeded_voice() {
        let text = "Hello. <|speaker:1|> Hi. <|speaker:3|> Hey.";
        assert_eq!(text_speakers(text), [0, 1, 3]);
        assert_eq!(text_speakers("<|speaker:2|> Hi. <|speaker:2|> Yo."), [2]);
        check_dialogue_references(DRAMA3, &[inline(1, "Hi.")], text).unwrap();
        check_unreferenced_speakers(DRAMA3, true, &[1], text).unwrap();
        check_unreferenced_speakers(DRAMA3, false, &[0, 1, 3], text).unwrap();
        let message = refused(check_unreferenced_speakers(DRAMA3, false, &[0, 1], text));
        assert!(message.contains("Speaker 3 has no reference"), "{message}");
    }

    #[test]
    fn references_are_trimmed_like_the_single_fields() {
        let mut references = [SpeechDialogueReference {
            speaker: Some(1),
            audio: Some(" UklGRg== ".into()),
            text: Some("  ".into()),
            saved_voice_id: Some(" voice ".into()),
        }];
        normalize_references(&mut references);
        assert_eq!(references[0].audio.as_deref(), Some("UklGRg=="));
        assert_eq!(references[0].text, None);
        assert_eq!(references[0].saved_voice_id.as_deref(), Some("voice"));
    }
}
