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
    pub max_tokens: Option<usize>,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
    pub word_timestamps: bool,
}

pub async fn execute(args: TranscribeArgs, server: &str) -> Result<()> {
    let TranscribeArgs {
        file,
        model,
        language,
        prompt,
        max_tokens,
        format,
        output,
        word_timestamps,
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
    for (name, value) in
        transcription_form_fields(model, language, prompt, max_tokens, format, word_timestamps)
    {
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

/// The text fields of the multipart transcription request.
fn transcription_form_fields(
    model: String,
    language: Option<String>,
    prompt: Option<String>,
    max_tokens: Option<usize>,
    format: TranscriptFormat,
    word_timestamps: bool,
) -> Vec<(&'static str, String)> {
    let format_str = if word_timestamps {
        "verbose_json"
    } else {
        match format {
            TranscriptFormat::Text => "text",
            TranscriptFormat::Json => "json",
            TranscriptFormat::VerboseJson => "verbose_json",
        }
    };

    let mut fields = vec![
        ("model", model),
        ("response_format", format_str.to_string()),
    ];

    if let Some(language) = language {
        fields.push(("language", language));
    }

    if let Some(prompt) = prompt
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        fields.push(("prompt", prompt));
    }

    if let Some(max_tokens) = max_tokens {
        fields.push(("max_tokens", max_tokens.to_string()));
    }

    if word_timestamps {
        fields.push(("timestamp_granularities[]", "word".to_string()));
    }

    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_fields_include_prompt_max_tokens_and_word_timestamps() {
        let fields = transcription_form_fields(
            "Granite-Speech-4.1-2B-Plus".to_string(),
            Some("en".to_string()),
            Some("keywords: izwi, granite".to_string()),
            Some(64),
            TranscriptFormat::Text,
            true,
        );
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
    }
}
