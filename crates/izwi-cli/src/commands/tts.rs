use crate::AudioFormat;
use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::path::PathBuf;

pub struct TtsArgs {
    pub text: String,
    pub model: String,
    pub speaker: Option<String>,
    pub saved_voice_id: Option<String>,
    pub reference_audio: Option<PathBuf>,
    pub reference_text: Option<String>,
    pub reference_text_file: Option<PathBuf>,
    pub instructions: Option<String>,
    pub output: Option<PathBuf>,
    pub format: AudioFormat,
    pub speed: f32,
    pub temperature: f32,
    pub stream: bool,
    pub allow_format_fallback: bool,
    pub play: bool,
}

pub async fn execute(args: TtsArgs, server: &str, theme: &Theme) -> Result<()> {
    let TtsArgs {
        text,
        model,
        speaker,
        saved_voice_id,
        reference_audio,
        reference_text,
        reference_text_file,
        instructions,
        output,
        format,
        speed,
        temperature,
        stream,
        allow_format_fallback,
        play,
    } = args;

    // Read text from stdin if "-"
    let text = if text == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| CliError::Io(e))?;
        buffer
    } else {
        text
    };

    if text.trim().is_empty() {
        return Err(CliError::InvalidInput("Text cannot be empty".to_string()));
    }
    reject_unsupported_cli_stream(&model, stream)?;

    theme.step(1, 2, &format!("Generating speech with '{}'...", model));

    let request_body = build_request_body(BuildRequestArgs {
        model,
        text,
        speaker,
        saved_voice_id,
        reference_audio,
        reference_text,
        reference_text_file,
        instructions,
        format: &format,
        speed,
        temperature,
        stream,
        allow_format_fallback,
    })?;

    let client = http::client(Some(std::time::Duration::from_secs(300)))?;

    let start_time = std::time::Instant::now();

    if stream {
        // Streaming mode
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );
        pb.set_message("Generating audio chunks...");

        let response = client
            .post(format!("{}/v1/audio/speech", server))
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

        pb.finish_with_message("Generation complete");

        let actual_format = response_header(response.headers(), "x-actual-response-format");
        let format_fallback = response_header(response.headers(), "x-response-format-fallback");

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| CliError::Other(e.to_string()))?;

        handle_output(
            audio_data,
            output.clone(),
            &format,
            actual_format.as_deref(),
            format_fallback.as_deref(),
            play,
            theme,
        )
        .await?;
    } else {
        // Non-streaming mode
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );

        pb.set_message("Sending request...");

        let response = client
            .post(format!("{}/v1/audio/speech", server))
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

        pb.set_message("Receiving audio...");

        let actual_format = response_header(response.headers(), "x-actual-response-format");
        let format_fallback = response_header(response.headers(), "x-response-format-fallback");

        let audio_data = response
            .bytes()
            .await
            .map_err(|e| CliError::Other(e.to_string()))?;

        pb.finish_with_message("Complete");

        handle_output(
            audio_data,
            output,
            &format,
            actual_format.as_deref(),
            format_fallback.as_deref(),
            play,
            theme,
        )
        .await?;
    }

    let duration = start_time.elapsed();
    theme.step(2, 2, &format!("Done in {:.2}s", duration.as_secs_f64()));

    Ok(())
}

async fn handle_output(
    audio_data: bytes::Bytes,
    output: Option<PathBuf>,
    format: &AudioFormat,
    actual_format: Option<&str>,
    format_fallback: Option<&str>,
    _play: bool,
    theme: &Theme,
) -> Result<()> {
    let output_was_explicit = output.is_some();
    let output_path = match output {
        Some(path) => path,
        None => {
            // Generate default filename
            let timestamp = chrono::Utc::now().timestamp();
            let extension = output_extension(format, actual_format);
            PathBuf::from(format!("izwi_output_{}.{}", timestamp, extension))
        }
    };

    if let Some(fallback) = format_fallback {
        let actual = actual_format.unwrap_or("wav");
        theme.warning(&format!(
            "Server used audio format fallback ({fallback}); saved {actual} audio."
        ));
        if output_was_explicit {
            theme.warning(
                "The output path was left unchanged; make sure its extension matches the saved audio.",
            );
        }
    }

    let mut file = tokio::fs::File::create(&output_path)
        .await
        .map_err(|e| CliError::Io(e))?;

    tokio::io::AsyncWriteExt::write_all(&mut file, &audio_data)
        .await
        .map_err(|e| CliError::Io(e))?;

    theme.success(&format!("Audio saved to: {}", output_path.display()));

    if _play {
        theme.info("Playing audio... (not implemented in this version)");
        // Would use rodio or external player here
    }

    Ok(())
}

fn response_header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn requested_format_label(format: &AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Ogg => "ogg",
        AudioFormat::Flac => "flac",
        AudioFormat::Aac => "aac",
    }
}

struct BuildRequestArgs<'a> {
    model: String,
    text: String,
    speaker: Option<String>,
    saved_voice_id: Option<String>,
    reference_audio: Option<PathBuf>,
    reference_text: Option<String>,
    reference_text_file: Option<PathBuf>,
    instructions: Option<String>,
    format: &'a AudioFormat,
    speed: f32,
    temperature: f32,
    stream: bool,
    allow_format_fallback: bool,
}

fn build_request_body(args: BuildRequestArgs<'_>) -> Result<Value> {
    let reference_text = resolve_reference_text(args.reference_text, args.reference_text_file)?;
    let has_reference_audio = args.reference_audio.is_some();
    let has_reference_text = reference_text.is_some();
    let has_saved_voice = normalize_optional_text(args.saved_voice_id.clone()).is_some();
    if has_saved_voice && (has_reference_audio || has_reference_text) {
        return Err(CliError::InvalidInput(
            "Use either --saved-voice-id or --reference-audio/--reference-text, not both."
                .to_string(),
        ));
    }
    if has_reference_audio != has_reference_text {
        return Err(CliError::InvalidInput(
            "Provide --reference-audio and --reference-text together.".to_string(),
        ));
    }

    let mut request_body = json!({
        "model": args.model,
        "input": args.text,
        "speed": args.speed,
        "temperature": args.temperature,
        "response_format": requested_format_label(args.format),
        "stream": args.stream,
        "allow_format_fallback": args.allow_format_fallback,
    });

    if let Some(speaker) = normalize_optional_text(args.speaker) {
        request_body["voice"] = Value::String(speaker);
    }
    if let Some(saved_voice_id) = normalize_optional_text(args.saved_voice_id) {
        request_body["saved_voice_id"] = Value::String(saved_voice_id);
    }
    if let Some(reference_audio) = args.reference_audio {
        let audio = fs::read(&reference_audio).map_err(CliError::Io)?;
        request_body["reference_audio"] = Value::String(STANDARD.encode(audio));
    }
    if let Some(reference_text) = reference_text {
        request_body["reference_text"] = Value::String(reference_text);
    }
    if let Some(instructions) = normalize_optional_text(args.instructions) {
        request_body["instructions"] = Value::String(instructions);
    }

    Ok(request_body)
}

fn resolve_reference_text(inline: Option<String>, file: Option<PathBuf>) -> Result<Option<String>> {
    let inline = normalize_optional_text(inline);
    let has_file = file.is_some();
    if inline.is_some() && has_file {
        return Err(CliError::InvalidInput(
            "Use either --reference-text or --reference-text-file, not both.".to_string(),
        ));
    }
    if let Some(path) = file {
        return Ok(normalize_optional_text(Some(
            fs::read_to_string(path).map_err(CliError::Io)?,
        )));
    }
    Ok(inline)
}

fn normalize_optional_text(raw: Option<String>) -> Option<String> {
    raw.map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn reject_unsupported_cli_stream(_model: &str, stream: bool) -> Result<()> {
    if !stream {
        return Ok(());
    }
    Err(CliError::InvalidInput(
        "`izwi tts --stream` receives Server-Sent Events that the CLI does not yet assemble into a playable audio file; omit --stream for CLI audio output."
            .to_string(),
    ))
}

fn output_extension(requested_format: &AudioFormat, actual_format: Option<&str>) -> &'static str {
    match actual_format.map(normalize_format_label).as_deref() {
        Some("wav") => "wav",
        Some("mp3") => "mp3",
        Some("ogg") => "ogg",
        Some("flac") => "flac",
        Some("aac") => "aac",
        Some("pcm_i16") => "pcm",
        Some("pcm_f32") => "f32",
        _ => requested_format_label(requested_format),
    }
}

fn normalize_format_label(label: &str) -> String {
    label.trim().to_ascii_lowercase().replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn output_extension_uses_actual_format_after_fallback() {
        assert_eq!(output_extension(&AudioFormat::Mp3, Some("wav")), "wav");
    }

    #[test]
    fn output_extension_falls_back_to_requested_format_when_header_missing() {
        assert_eq!(output_extension(&AudioFormat::Flac, None), "flac");
    }

    #[test]
    fn output_extension_handles_raw_audio_labels() {
        assert_eq!(output_extension(&AudioFormat::Wav, Some("pcm-f32")), "f32");
    }

    #[test]
    fn cli_streaming_is_rejected_before_request() {
        let err = reject_unsupported_cli_stream("Kokoro-82M", true)
            .expect_err("streaming should be rejected");

        assert!(err.to_string().contains("omit --stream"));
        reject_unsupported_cli_stream("Kokoro-82M", false).expect("non-streaming is supported");
    }

    #[test]
    fn request_body_includes_saved_voice_without_default_voice() {
        let body = build_request_body(BuildRequestArgs {
            model: "VibeVoice-1.5B".to_string(),
            text: "hello".to_string(),
            speaker: None,
            saved_voice_id: Some(" voice-1 ".to_string()),
            reference_audio: None,
            reference_text: None,
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: 1.0,
            temperature: 0.7,
            stream: false,
            allow_format_fallback: false,
        })
        .expect("request body");

        assert!(body.get("voice").is_none());
        assert_eq!(body["saved_voice_id"], "voice-1");
    }

    #[test]
    fn request_body_encodes_direct_reference_voice_pair() {
        let dir = tempdir().expect("temp dir");
        let audio_path = dir.path().join("voice.wav");
        fs::write(&audio_path, b"RIFF").expect("audio fixture");

        let body = build_request_body(BuildRequestArgs {
            model: "VibeVoice-1.5B".to_string(),
            text: "hello".to_string(),
            speaker: Some("Speaker 1".to_string()),
            saved_voice_id: None,
            reference_audio: Some(audio_path),
            reference_text: Some(" reference words ".to_string()),
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: 1.0,
            temperature: 0.7,
            stream: false,
            allow_format_fallback: false,
        })
        .expect("request body");

        assert_eq!(body["voice"], "Speaker 1");
        assert_eq!(body["reference_audio"], "UklGRg==");
        assert_eq!(body["reference_text"], "reference words");
    }

    #[test]
    fn request_body_rejects_partial_direct_reference() {
        let err = build_request_body(BuildRequestArgs {
            model: "VibeVoice-1.5B".to_string(),
            text: "hello".to_string(),
            speaker: None,
            saved_voice_id: None,
            reference_audio: None,
            reference_text: Some("reference words".to_string()),
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: 1.0,
            temperature: 0.7,
            stream: false,
            allow_format_fallback: false,
        })
        .expect_err("partial reference should fail");

        assert!(err.to_string().contains("Provide --reference-audio"));
    }
}
