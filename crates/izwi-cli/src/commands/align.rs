use crate::error::{CliError, Result};
use crate::http;
use crate::TranscriptFormat;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::path::PathBuf;

pub struct AlignArgs {
    pub file: PathBuf,
    pub text: String,
    pub model: String,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
}

pub async fn execute(args: AlignArgs, server: &str) -> Result<()> {
    let AlignArgs {
        file,
        text,
        model,
        format,
        output,
    } = args;

    // Verify file exists
    if !file.exists() {
        return Err(CliError::InvalidInput(format!(
            "File not found: {}",
            file.display()
        )));
    }

    if text.trim().is_empty() {
        return Err(CliError::InvalidInput(
            "Reference text cannot be empty".to_string(),
        ));
    }

    // Read audio file
    let audio_data = tokio::fs::read(&file).await.map_err(CliError::Io)?;
    let audio_base64 = STANDARD.encode(&audio_data);

    let format_str = match format {
        TranscriptFormat::Text => "text",
        TranscriptFormat::Json => "json",
        TranscriptFormat::VerboseJson => "verbose_json",
    };

    let request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
        "text": text,
        "response_format": format_str,
    });

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let response = client
        .post(format!("{}/v1/audio/align", server))
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
            .map_err(CliError::Io)?;
        println!("Alignment saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}
