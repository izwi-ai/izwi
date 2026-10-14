use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::AudioFormat;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::HeaderMap;
use serde_json::{json, Value};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

const TTS_REQUEST_TIMEOUT: Duration = Duration::from_secs(900);

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
    pub speed: Option<f32>,
    pub temperature: Option<f32>,
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
            .map_err(CliError::Io)?;
        buffer
    } else {
        text
    };

    if text.trim().is_empty() {
        return Err(CliError::InvalidInput("Text cannot be empty".to_string()));
    }
    reject_unsupported_cli_stream(&format, stream)?;

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

    // Dense diffusion TTS can legitimately exceed five minutes on
    // resource-constrained local Metal devices. Keep a finite client fence,
    // but leave enough room for the server's own configured request deadline.
    let client = http::client(Some(TTS_REQUEST_TIMEOUT))?;

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

        // Server-Sent Events: `audio.chunk` events carry base64 16-bit PCM,
        // assembled here into one WAV file.
        let mut assembler = SseSpeechAssembler::default();
        let mut body = response.bytes_stream();
        while let Some(bytes) = futures::StreamExt::next(&mut body).await {
            let bytes = bytes.map_err(|e| CliError::Other(e.to_string()))?;
            if assembler.push(&bytes)? && assembler.first_audio.is_none() {
                assembler.first_audio = Some(start_time.elapsed());
                pb.set_message("Receiving audio...");
            }
        }
        let audio_data = assembler.finish()?;
        pb.finish_with_message("Generation complete");
        if let Some(first_audio) = assembler.first_audio {
            theme.info(&format!(
                "First audio after {:.2}s",
                first_audio.as_secs_f64()
            ));
        }

        handle_output(
            audio_data,
            output.clone(),
            &format,
            Some("wav"),
            None,
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
        .map_err(CliError::Io)?;

    tokio::io::AsyncWriteExt::write_all(&mut file, &audio_data)
        .await
        .map_err(CliError::Io)?;

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
    speed: Option<f32>,
    temperature: Option<f32>,
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
        // A stream carries raw 16-bit PCM chunks, assembled into a WAV.
        "response_format": if args.stream { "pcm" } else { requested_format_label(args.format) },
        "stream": args.stream,
        "allow_format_fallback": args.allow_format_fallback,
    });
    // Unset sampling and speed keep the model's own defaults.
    if let Some(speed) = args.speed {
        request_body["speed"] = json!(speed);
    }
    if let Some(temperature) = args.temperature {
        request_body["temperature"] = json!(temperature);
    }

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

/// `--stream` writes WAV: the stream carries raw PCM chunks.
fn reject_unsupported_cli_stream(format: &AudioFormat, stream: bool) -> Result<()> {
    if !stream || matches!(format, AudioFormat::Wav) {
        return Ok(());
    }
    Err(CliError::InvalidInput(
        "`izwi tts --stream` writes WAV audio; use --format wav or omit --stream.".to_string(),
    ))
}

/// Assembles an OpenAI-route speech SSE stream (`audio.started`,
/// `audio.chunk` with base64 16-bit PCM, `audio.done`, `audio.failed`) into
/// a WAV file.
#[derive(Debug, Default)]
struct SseSpeechAssembler {
    pending: String,
    pcm: Vec<u8>,
    sample_rate: Option<u32>,
    done: bool,
    first_audio: Option<Duration>,
}

impl SseSpeechAssembler {
    /// Feed received bytes; returns whether they carried audio.
    fn push(&mut self, bytes: &[u8]) -> Result<bool> {
        self.pending.push_str(&String::from_utf8_lossy(bytes));
        let mut audio = false;
        while let Some(end) = self.pending.find('\n') {
            let line: String = self.pending.drain(..=end).collect();
            let Some(data) = line.trim_end().strip_prefix("data:") else {
                continue;
            };
            let event: Value = serde_json::from_str(data.trim())
                .map_err(|e| CliError::Other(format!("Malformed speech stream event: {e}")))?;
            match event["event"].as_str() {
                Some("audio.started") => {
                    self.sample_rate = event["sample_rate"].as_u64().map(|rate| rate as u32);
                }
                Some("audio.chunk") => {
                    if let Some(rate) = event["sample_rate"].as_u64() {
                        self.sample_rate.get_or_insert(rate as u32);
                    }
                    if let Some(encoded) = event["audio_base64"].as_str() {
                        let bytes = STANDARD
                            .decode(encoded)
                            .map_err(|e| CliError::Other(format!("Malformed audio chunk: {e}")))?;
                        audio |= !bytes.is_empty();
                        self.pcm.extend_from_slice(&bytes);
                    }
                }
                Some("audio.done") => self.done = true,
                Some("audio.failed") => {
                    return Err(CliError::Other(format!(
                        "Speech generation failed: {}",
                        event["error"].as_str().unwrap_or("unknown error")
                    )))
                }
                _ => {}
            }
        }
        Ok(audio)
    }

    /// The assembled audio as a 16-bit mono WAV.
    fn finish(&self) -> Result<bytes::Bytes> {
        if !self.done {
            return Err(CliError::Other(
                "Speech stream ended before audio.done".to_string(),
            ));
        }
        let rate = self
            .sample_rate
            .filter(|rate| *rate > 0)
            .ok_or_else(|| CliError::Other("Speech stream carried no audio".to_string()))?;
        let data_len = u32::try_from(self.pcm.len())
            .map_err(|_| CliError::Other("Speech stream is too long for WAV".to_string()))?;
        let mut wav = Vec::with_capacity(44 + self.pcm.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&rate.to_le_bytes());
        wav.extend_from_slice(&(rate * 2).to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(&self.pcm);
        Ok(bytes::Bytes::from(wav))
    }
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
    fn cli_streaming_writes_wav_only() {
        reject_unsupported_cli_stream(&AudioFormat::Wav, true).expect("WAV streams");
        reject_unsupported_cli_stream(&AudioFormat::Mp3, false).expect("non-stream MP3");
        let err = reject_unsupported_cli_stream(&AudioFormat::Mp3, true)
            .expect_err("a compressed stream is refused");
        assert!(err.to_string().contains("--format wav"));
    }

    /// TPIPE-25: `--stream` assembles the SSE PCM chunks into a WAV, even
    /// when an event is split across network reads.
    #[test]
    fn sse_chunks_assemble_into_a_wav() {
        let pcm = [1u8, 0, 2, 0, 3, 0];
        let events = format!(
            "data: {{\"event\":\"audio.started\",\"sample_rate\":24000}}\n\n\
             data: {{\"event\":\"audio.chunk\",\"sequence\":0,\"audio_base64\":\"{}\"}}\n\n\
             : keepalive\n\n\
             data: {{\"event\":\"audio.done\"}}\n\n",
            STANDARD.encode(pcm)
        );
        let mut assembler = SseSpeechAssembler::default();
        let (head, tail) = events.as_bytes().split_at(70);
        assembler.push(head).unwrap();
        assembler.push(tail).unwrap();
        let wav = assembler.finish().unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(&wav[44..], &pcm);

        let mut failed = SseSpeechAssembler::default();
        let error = failed
            .push(b"data: {\"event\":\"audio.failed\",\"error\":\"busy\"}\n")
            .unwrap_err();
        assert!(error.to_string().contains("busy"));
        assert!(SseSpeechAssembler::default().finish().is_err());
    }

    #[test]
    fn unset_sampling_and_speed_are_not_sent() {
        let body = build_request_body(BuildRequestArgs {
            model: "Kokoro-82M".to_string(),
            text: "hello".to_string(),
            speaker: None,
            saved_voice_id: None,
            reference_audio: None,
            reference_text: None,
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: None,
            temperature: None,
            stream: true,
            allow_format_fallback: false,
        })
        .expect("request body");
        assert!(body.get("speed").is_none() && body.get("temperature").is_none());
        assert_eq!(body["response_format"], "pcm");
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
            speed: Some(1.0),
            temperature: Some(0.7),
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
            speed: Some(1.0),
            temperature: Some(0.7),
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
            speed: Some(1.0),
            temperature: Some(0.7),
            stream: false,
            allow_format_fallback: false,
        })
        .expect_err("partial reference should fail");

        assert!(err.to_string().contains("Provide --reference-audio"));
    }
}
