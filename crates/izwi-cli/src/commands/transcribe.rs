use crate::error::{CliError, Result};
use crate::http;
use crate::utils;
use crate::TranscriptFormat;
use reqwest::multipart::Form;
use std::path::PathBuf;

pub struct TranscribeArgs {
    pub file: PathBuf,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub translate_to: Option<String>,
    pub keywords: Vec<String>,
    pub max_tokens: Option<usize>,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
    pub word_timestamps: bool,
    pub speaker_labels: bool,
}

pub async fn execute(args: TranscribeArgs, server: &str) -> Result<()> {
    let TranscribeArgs {
        file,
        model,
        language,
        prompt,
        translate_to,
        keywords,
        max_tokens,
        format,
        output,
        word_timestamps,
        speaker_labels,
    } = args;

    // Verify file exists
    if !file.exists() {
        return Err(CliError::InvalidInput(format!(
            "File not found: {}",
            file.display()
        )));
    }

    // The audio goes as a multipart file, not base64 JSON, so the whole
    // upload limit is available for the audio itself.
    let mut form = Form::new().part("file", utils::audio_file_part(&file).await?);
    let fields = TranscriptionFormFields {
        model,
        language,
        prompt,
        translate_to,
        keywords,
        max_tokens,
        format,
        word_timestamps,
        speaker_labels,
    };
    for (name, value) in fields.into_form_fields() {
        form = form.text(name, value);
    }

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let response = client
        .post(format!("{}/v1/audio/transcriptions", server))
        .multipart(form)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }

    let result = response.text().await?;

    // Output result
    if let Some(output_path) = output {
        tokio::fs::write(&output_path, result)
            .await
            .map_err(CliError::Io)?;
        println!("Transcription saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}

/// What the multipart transcription request carries besides the audio.
struct TranscriptionFormFields {
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    translate_to: Option<String>,
    keywords: Vec<String>,
    max_tokens: Option<usize>,
    format: TranscriptFormat,
    word_timestamps: bool,
    speaker_labels: bool,
}

impl TranscriptionFormFields {
    /// The text fields of the multipart transcription request.
    fn into_form_fields(self) -> Vec<(&'static str, String)> {
        // Word timestamps and speaker turns come back in verbose_json only.
        let format_str = if self.word_timestamps || self.speaker_labels {
            "verbose_json"
        } else {
            match self.format {
                TranscriptFormat::Text => "text",
                TranscriptFormat::Json => "json",
                TranscriptFormat::VerboseJson => "verbose_json",
            }
        };

        let mut fields = vec![
            ("model", self.model),
            ("response_format", format_str.to_string()),
        ];

        if let Some(language) = self.language {
            fields.push(("language", language));
        }

        if let Some(prompt) = non_empty(self.prompt) {
            fields.push(("prompt", prompt));
        }

        if let Some(target) = non_empty(self.translate_to) {
            fields.push(("target_language", target));
        }

        // One field per keyword. The server also splits each field on
        // commas and newlines, so a keyword cannot hold a comma.
        for keyword in self.keywords {
            if let Some(keyword) = non_empty(Some(keyword)) {
                fields.push(("keywords", keyword));
            }
        }

        if let Some(max_tokens) = self.max_tokens {
            fields.push(("max_tokens", max_tokens.to_string()));
        }

        if self.word_timestamps {
            fields.push(("timestamp_granularities[]", "word".to_string()));
        }

        if self.speaker_labels {
            fields.push(("speaker_labels", "true".to_string()));
        }

        fields
    }
}

fn non_empty(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_labels_ask_for_verbose_json_and_the_extension_field() {
        let fields = TranscriptionFormFields {
            model: "Nemotron-3.5-Transcribe-30B-A4B-FP8".to_string(),
            language: None,
            prompt: None,
            translate_to: None,
            keywords: Vec::new(),
            max_tokens: None,
            format: TranscriptFormat::Text,
            word_timestamps: false,
            speaker_labels: true,
        }
        .into_form_fields();
        assert!(fields.contains(&("response_format", "verbose_json".to_string())));
        assert!(fields.contains(&("speaker_labels", "true".to_string())));
        assert!(!fields
            .iter()
            .any(|(name, _)| *name == "timestamp_granularities[]"));
    }

    #[test]
    fn form_fields_include_prompt_max_tokens_and_word_timestamps() {
        let fields = TranscriptionFormFields {
            model: "Granite-Speech-4.1-2B-Plus".to_string(),
            language: Some("en".to_string()),
            prompt: Some("keywords: izwi, granite".to_string()),
            translate_to: None,
            keywords: Vec::new(),
            max_tokens: Some(64),
            format: TranscriptFormat::Text,
            word_timestamps: true,
            speaker_labels: false,
        }
        .into_form_fields();
        let field = |name: &str| {
            fields
                .iter()
                .find(|(field, _)| *field == name)
                .map(|(_, value)| value.as_str())
        };

        assert_eq!(field("model"), Some("Granite-Speech-4.1-2B-Plus"));
        assert_eq!(field("response_format"), Some("verbose_json"));
        assert_eq!(field("language"), Some("en"));
        assert_eq!(field("prompt"), Some("keywords: izwi, granite"));
        assert_eq!(field("max_tokens"), Some("64"));
        assert_eq!(field("timestamp_granularities[]"), Some("word"));
        assert_eq!(field("audio_base64"), None);
        assert_eq!(field("target_language"), None);
        assert_eq!(field("keywords"), None);
    }

    #[test]
    fn form_fields_carry_a_translation_target_and_one_field_per_keyword() {
        let fields = TranscriptionFormFields {
            model: "Granite-Speech-5.0-2B".to_string(),
            language: Some("de".to_string()),
            prompt: None,
            translate_to: Some(" en ".to_string()),
            keywords: vec![
                "Izwi".to_string(),
                " ".to_string(),
                "Granite Speech".to_string(),
            ],
            max_tokens: None,
            format: TranscriptFormat::Text,
            word_timestamps: false,
            speaker_labels: false,
        }
        .into_form_fields();

        assert!(fields.contains(&("target_language", "en".to_string())));
        let keywords = fields
            .iter()
            .filter(|(name, _)| *name == "keywords")
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(keywords, vec!["Izwi", "Granite Speech"]);
    }
}
