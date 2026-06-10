use crate::error::{CliError, Result};
use crate::http;
use crate::TranscriptFormat;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
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

    // Read audio file
    let audio_data = tokio::fs::read(&file).await.map_err(CliError::Io)?;
    let audio_base64 = STANDARD.encode(&audio_data);

    let request_body = build_transcription_request_body(
        model,
        audio_base64,
        language,
        prompt,
        max_tokens,
        format,
        word_timestamps,
    );

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let response = client
        .post(format!("{}/v1/audio/transcriptions", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let result = response
        .text()
        .await
        .map_err(|e| CliError::Other(e.to_string()))?;

    // Output result
    if let Some(output_path) = output {
        tokio::fs::write(&output_path, result)
            .await
            .map_err(|e| CliError::Io(e))?;
        println!("Transcription saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}

fn build_transcription_request_body(
    model: String,
    audio_base64: String,
    language: Option<String>,
    prompt: Option<String>,
    max_tokens: Option<usize>,
    format: TranscriptFormat,
    word_timestamps: bool,
) -> serde_json::Value {
    let format_str = if word_timestamps {
        "verbose_json"
    } else {
        match format {
            TranscriptFormat::Text => "text",
            TranscriptFormat::Json => "json",
            TranscriptFormat::VerboseJson => "verbose_json",
        }
    };

    let mut request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
        "response_format": format_str,
    });

    if let Some(language) = language {
        request_body["language"] = serde_json::Value::String(language);
    }

    if let Some(prompt) = prompt
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        request_body["prompt"] = serde_json::Value::String(prompt);
    }

    if let Some(max_tokens) = max_tokens {
        request_body["max_tokens"] = serde_json::json!(max_tokens);
    }

    if word_timestamps {
        request_body["timestamp_granularities"] = serde_json::json!(["word"]);
    }

    request_body
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_includes_prompt_max_tokens_and_word_timestamps() {
        let body = build_transcription_request_body(
            "Granite-Speech-4.1-2B-Plus".to_string(),
            "AAAA".to_string(),
            Some("en".to_string()),
            Some("keywords: izwi, granite".to_string()),
            Some(64),
            TranscriptFormat::Text,
            true,
        );

        assert_eq!(body["model"], "Granite-Speech-4.1-2B-Plus");
        assert_eq!(body["response_format"], "verbose_json");
        assert_eq!(body["language"], "en");
        assert_eq!(body["prompt"], "keywords: izwi, granite");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["timestamp_granularities"][0], "word");
    }
}
