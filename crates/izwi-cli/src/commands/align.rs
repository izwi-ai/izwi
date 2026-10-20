use crate::error::{CliError, Result};
use crate::http;
use crate::utils;
use crate::TranscriptFormat;
use reqwest::multipart::Form;
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

    let format_str = match format {
        TranscriptFormat::Text => "text",
        TranscriptFormat::Json => "json",
        TranscriptFormat::VerboseJson => "verbose_json",
    };

    // The audio goes as a multipart file, not base64 JSON, so the whole
    // upload limit is available for the audio itself.
    let form = Form::new()
        .part("file", utils::audio_file_part(&file).await?)
        .text("model", model)
        .text("text", text)
        .text("response_format", format_str);

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let response = client
        .post(format!("{}/v1/audio/align", server))
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
        println!("Alignment saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}
