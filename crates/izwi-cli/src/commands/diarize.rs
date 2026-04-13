use crate::error::{CliError, Result};
use crate::http;
use crate::TranscriptFormat;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use std::path::PathBuf;

pub struct DiarizeArgs {
    pub file: PathBuf,
    pub model: String,
    pub num_speakers: Option<u32>,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
    pub transcribe: bool,
    pub asr_model: String,
}

pub async fn execute(args: DiarizeArgs, server: &str) -> Result<()> {
    let DiarizeArgs {
        file,
        model,
        num_speakers,
        format,
        output,
        transcribe,
        asr_model,
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

    let format_str = match format {
        TranscriptFormat::Text => "text",
        TranscriptFormat::Json => "json",
        TranscriptFormat::VerboseJson => "verbose_json",
    };

    if transcribe {
        eprintln!(
            "Note: --transcribe is now a compatibility flag; diarization responses already include transcript output."
        );
    }

    let mut request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
        "response_format": format_str,
        "asr_model": asr_model,
    });

    if let Some(num) = num_speakers {
        request_body["min_speakers"] = serde_json::Value::Number(num.into());
        request_body["max_speakers"] = serde_json::Value::Number(num.into());
    }

    let client = http::client(Some(std::time::Duration::from_secs(600)))?;
    let response = client
        .post(format!("{}/v1/audio/diarizations", server))
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
        println!("Diarization saved to: {}", output_path.display());
    } else {
        println!("{}", result);
    }

    Ok(())
}
