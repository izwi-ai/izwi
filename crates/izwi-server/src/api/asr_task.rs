//! Translation and keyword fields of the transcription routes.
//!
//! `target_language` asks a model that translates speech (Granite Speech
//! 5.0) for a translation instead of a transcript. `keywords` biases
//! recognition toward names and terms: a JSON list, or text with one keyword
//! per line or comma, in any number of form fields.

use izwi_core::AsrTaskOptions;

use crate::error::ApiError;

/// A JSON `keywords` value: a list, or one comma- or newline-separated string.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum KeywordsField {
    List(Vec<String>),
    Text(String),
}

impl KeywordsField {
    pub(crate) fn into_keywords(self) -> Vec<String> {
        match self {
            Self::List(keywords) => keywords,
            Self::Text(text) => split_keywords(&text),
        }
    }
}

/// The keywords of one form field: one per line or comma, blanks dropped.
pub(crate) fn split_keywords(text: &str) -> Vec<String> {
    text.split([',', '\n'])
        .map(str::trim)
        .filter(|keyword| !keyword.is_empty())
        .map(str::to_string)
        .collect()
}

/// Reject a translation or keyword list the routed model cannot honour, and
/// word timings for a translation (they would time the target-language text
/// against the source speech).
pub(crate) fn validate_asr_task_request(
    model_id: Option<&str>,
    language: Option<&str>,
    prompt: Option<&str>,
    task: &AsrTaskOptions,
    timestamps_requested: bool,
) -> Result<(), ApiError> {
    if task.translate_to.is_some() && timestamps_requested {
        return Err(ApiError::bad_request(
            "Word timestamps are not available for a translation; omit `target_language` or the timestamps",
        ));
    }
    izwi_core::validate_asr_task(
        izwi_core::resolve_asr_model_variant(model_id),
        language,
        prompt,
        task,
    )
    .map_err(|err| ApiError::bad_request(err.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_split_on_commas_and_lines() {
        assert_eq!(
            split_keywords(" Izwi, Granite\nAgentic AI ,,\n "),
            vec!["Izwi", "Granite", "Agentic AI"]
        );
        let list: KeywordsField = serde_json::from_str(r#"["Smith, John", "Izwi"]"#).unwrap();
        assert_eq!(list.into_keywords(), vec!["Smith, John", "Izwi"]);
        let text: KeywordsField = serde_json::from_str(r#""Izwi, Granite""#).unwrap();
        assert_eq!(text.into_keywords(), vec!["Izwi", "Granite"]);
    }

    #[test]
    fn a_translation_needs_a_translating_model_and_no_timestamps() {
        let translate = AsrTaskOptions::new(Some("English".to_string()), Vec::new());
        validate_asr_task_request(
            Some("Granite-Speech-5.0-2B"),
            Some("fr"),
            None,
            &translate,
            false,
        )
        .unwrap();
        assert!(validate_asr_task_request(
            Some("Granite-Speech-5.0-2B"),
            Some("fr"),
            None,
            &translate,
            true,
        )
        .is_err());
        assert!(validate_asr_task_request(
            Some("Granite-Speech-4.1-2B"),
            Some("fr"),
            None,
            &translate,
            false,
        )
        .is_err());
        let keywords = AsrTaskOptions::new(None, vec!["Izwi".to_string()]);
        validate_asr_task_request(Some("Granite-Speech-5.0-2B"), None, None, &keywords, true)
            .unwrap();
        assert!(validate_asr_task_request(
            Some("Whisper-Large-v3-Turbo"),
            None,
            None,
            &keywords,
            false
        )
        .is_err());
    }
}
