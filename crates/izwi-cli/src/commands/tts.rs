use crate::error::{CliError, Result};
use crate::http;
use crate::style::{self, Theme};
use crate::AudioFormat;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::StreamExt;
use indicatif::ProgressBar;
use serde_json::{json, Value};
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};

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
    pub language: Option<String>,
    pub reference_language: Option<String>,
    pub exaggeration: Option<f32>,
    pub cfg_weight: Option<f32>,
    pub min_p: Option<f32>,
    pub stream: bool,
    pub play: bool,
}

/// Model-specific speech controls; each is sent only when set, and the
/// server refuses one the model does not support.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct SpeechControls {
    exaggeration: Option<f32>,
    cfg_weight: Option<f32>,
    min_p: Option<f32>,
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
        language,
        reference_language,
        exaggeration,
        cfg_weight,
        min_p,
        stream,
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
    if play && matches!(format, AudioFormat::Pcm) {
        return Err(CliError::InvalidInput(
            "--play needs WAV audio; drop --format pcm.".to_string(),
        ));
    }

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
        language,
        reference_language,
        controls: SpeechControls {
            exaggeration,
            cfg_weight,
            min_p,
        },
        stream,
    })?;

    // Dense diffusion TTS can legitimately exceed five minutes on
    // resource-constrained local Metal devices. Keep a finite client fence,
    // but leave enough room for the server's own configured request deadline.
    let client = http::client(Some(TTS_REQUEST_TIMEOUT))?;

    let start_time = Instant::now();
    let progress = style::spinner(if stream {
        "Generating audio chunks..."
    } else {
        "Generating audio..."
    });
    let generated = generate_audio(
        &client,
        server,
        &request_body,
        stream,
        &progress,
        start_time,
    )
    .await;
    progress.finish_and_clear();
    let (audio_data, first_audio) = generated?;
    if let Some(first_audio) = first_audio {
        theme.info(&format!(
            "First audio after {:.2}s",
            first_audio.as_secs_f64()
        ));
    }

    let output_path = save_audio(audio_data, output, &format, theme).await?;
    let duration = start_time.elapsed();
    theme.step(2, 2, &format!("Done in {:.2}s", duration.as_secs_f64()));

    if play {
        play_audio(&output_path, theme).await?;
    }

    Ok(())
}

/// The generated audio, and for a stream when its first audio arrived.
async fn generate_audio(
    client: &reqwest::Client,
    server: &str,
    request_body: &Value,
    stream: bool,
    progress: &ProgressBar,
    started: Instant,
) -> Result<(bytes::Bytes, Option<Duration>)> {
    let response = client
        .post(format!("{}/v1/audio/speech", server))
        .json(request_body)
        .send()
        .await?;
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    if !stream {
        progress.set_message("Receiving audio...");
        return Ok((response.bytes().await?, None));
    }

    // Server-Sent Events: `audio.chunk` events carry base64 16-bit PCM,
    // assembled here into one WAV file.
    let mut assembler = SseSpeechAssembler::default();
    let mut body = response.bytes_stream();
    while let Some(bytes) = body.next().await {
        if assembler.push(&bytes?)? && assembler.first_audio.is_none() {
            assembler.first_audio = Some(started.elapsed());
            progress.set_message("Receiving audio...");
        }
    }
    Ok((assembler.finish()?, assembler.first_audio))
}

async fn save_audio(
    audio_data: bytes::Bytes,
    output: Option<PathBuf>,
    format: &AudioFormat,
    theme: &Theme,
) -> Result<PathBuf> {
    let output_path = output.unwrap_or_else(|| {
        let timestamp = chrono::Utc::now().timestamp();
        PathBuf::from(format!(
            "izwi_output_{}.{}",
            timestamp,
            format_label(format)
        ))
    });
    tokio::fs::write(&output_path, &audio_data)
        .await
        .map_err(|error| {
            CliError::Other(format!(
                "Could not write {}: {error}",
                output_path.display()
            ))
        })?;
    theme.success(&format!("Audio saved to: {}", output_path.display()));
    Ok(output_path)
}

/// Plays a saved WAV with the first command-line player the system has.
async fn play_audio(path: &Path, theme: &Theme) -> Result<()> {
    theme.info("Playing audio...");
    let players = audio_players(path);
    for (program, args) in &players {
        let played = tokio::process::Command::new(program)
            .args(args)
            .stdin(Stdio::null())
            .status()
            .await;
        match played {
            Ok(status) if status.success() => return Ok(()),
            Ok(status) => {
                return Err(CliError::Other(format!(
                    "{program} could not play {} ({status})",
                    path.display()
                )))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(CliError::Other(format!("Could not run {program}: {error}"))),
        }
    }
    let tried: Vec<&str> = players.iter().map(|(program, _)| *program).collect();
    Err(CliError::Other(format!(
        "No audio player found (tried {}); the audio is saved at {}.",
        tried.join(", "),
        path.display()
    )))
}

/// The players `--play` tries, in order, with their arguments.
fn audio_players(path: &Path) -> Vec<(&'static str, Vec<OsString>)> {
    let path = OsString::from(path);
    if cfg!(target_os = "macos") {
        vec![("afplay", vec![path])]
    } else if cfg!(windows) {
        let path = std::path::absolute(&path).unwrap_or_else(|_| PathBuf::from(&path));
        let script = format!(
            "(New-Object Media.SoundPlayer '{}').PlaySync()",
            path.display().to_string().replace('\'', "''")
        );
        let args = ["-NoProfile", "-NonInteractive", "-Command"];
        let mut args: Vec<OsString> = args.into_iter().map(OsString::from).collect();
        args.push(script.into());
        vec![("powershell", args)]
    } else {
        let with = |flags: &[&str]| {
            let mut args: Vec<OsString> = flags.iter().map(OsString::from).collect();
            args.push(path.clone());
            args
        };
        vec![
            ("paplay", with(&[])),
            ("aplay", with(&["-q"])),
            (
                "ffplay",
                with(&["-nodisp", "-autoexit", "-loglevel", "quiet"]),
            ),
            ("play", with(&["-q"])),
        ]
    }
}

/// The format's name in requests and its file extension.
fn format_label(format: &AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "wav",
        AudioFormat::Pcm => "pcm",
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
    language: Option<String>,
    reference_language: Option<String>,
    controls: SpeechControls,
    stream: bool,
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
    // Whether the transcript is needed too is the model's call (Chatterbox
    // clones from audio alone); the server says which.
    if has_reference_text && !has_reference_audio {
        return Err(CliError::InvalidInput(
            "--reference-text needs --reference-audio.".to_string(),
        ));
    }

    let mut request_body = json!({
        "model": args.model,
        "input": args.text,
        // A stream carries raw 16-bit PCM chunks, assembled into a WAV.
        "response_format": if args.stream { "pcm" } else { format_label(args.format) },
        "stream": args.stream,
    });
    // Unset sampling and speed keep the model's own defaults.
    if let Some(speed) = args.speed {
        request_body["speed"] = json!(speed);
    }
    if let Some(temperature) = args.temperature {
        request_body["temperature"] = json!(temperature);
    }
    if let Some(language) = normalize_optional_text(args.language) {
        request_body["language"] = Value::String(language);
    }
    if let Some(language) = normalize_optional_text(args.reference_language) {
        request_body["reference_language"] = Value::String(language);
    }
    let controls = args.controls;
    for (name, value) in [
        ("exaggeration", controls.exaggeration),
        ("cfg_weight", controls.cfg_weight),
        ("min_p", controls.min_p),
    ] {
        if let Some(value) = value {
            request_body[name] = json!(value);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn cli_streaming_writes_wav_only() {
        reject_unsupported_cli_stream(&AudioFormat::Wav, true).expect("WAV streams");
        reject_unsupported_cli_stream(&AudioFormat::Pcm, false).expect("non-stream PCM");
        let err = reject_unsupported_cli_stream(&AudioFormat::Pcm, true)
            .expect_err("a raw PCM stream file is refused");
        assert!(err.to_string().contains("--format wav"));
    }

    #[test]
    fn requests_name_formats_the_server_encodes() {
        let body = |format: &AudioFormat| {
            build_request_body(BuildRequestArgs {
                model: "Kokoro-82M".to_string(),
                text: "hello".to_string(),
                speaker: None,
                saved_voice_id: None,
                reference_audio: None,
                reference_text: None,
                reference_text_file: None,
                instructions: None,
                format,
                speed: None,
                temperature: None,
                language: None,
                reference_language: None,
                controls: SpeechControls::default(),
                stream: false,
            })
            .expect("request body")
        };
        assert_eq!(body(&AudioFormat::Wav)["response_format"], "wav");
        assert_eq!(body(&AudioFormat::Pcm)["response_format"], "pcm");
        assert!(body(&AudioFormat::Wav)
            .get("allow_format_fallback")
            .is_none());
    }

    #[test]
    fn play_tries_a_system_player_with_the_file() {
        let players = audio_players(Path::new("speech.wav"));
        assert!(!players.is_empty());
        for (program, args) in players {
            let last = args.last().expect("player arguments").to_string_lossy();
            assert!(last.contains("speech.wav"), "{program}: {last}");
        }
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
            language: None,
            reference_language: None,
            controls: SpeechControls::default(),
            stream: true,
        })
        .expect("request body");
        assert!(body.get("speed").is_none() && body.get("temperature").is_none());
        assert_eq!(body["response_format"], "pcm");
    }

    #[test]
    fn speech_controls_are_sent_only_when_set() {
        let body = build_request_body(BuildRequestArgs {
            model: "Chatterbox-500M".to_string(),
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
            language: None,
            reference_language: None,
            controls: SpeechControls {
                exaggeration: Some(0.8),
                cfg_weight: Some(0.0),
                min_p: None,
            },
            stream: false,
        })
        .expect("request body");
        assert_eq!(body["exaggeration"], json!(0.8f32));
        assert_eq!(body["cfg_weight"], json!(0.0f32));
        assert!(body.get("min_p").is_none());
    }

    #[test]
    fn a_saved_voice_carries_a_cross_lingual_reference_language() {
        let body = build_request_body(BuildRequestArgs {
            model: "Maya2-Native-2.5B".to_string(),
            text: "नमस्ते".to_string(),
            speaker: None,
            saved_voice_id: Some("voice-1".to_string()),
            reference_audio: None,
            reference_text: None,
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: None,
            temperature: None,
            language: Some("hi".to_string()),
            reference_language: Some(" en ".to_string()),
            controls: SpeechControls::default(),
            stream: false,
        })
        .expect("request body");
        assert_eq!(body["language"], "hi");
        assert_eq!(body["reference_language"], "en");
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
            language: None,
            reference_language: None,
            controls: SpeechControls::default(),
            stream: false,
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
            language: None,
            reference_language: None,
            controls: SpeechControls::default(),
            stream: false,
        })
        .expect("request body");

        assert_eq!(body["voice"], "Speaker 1");
        assert_eq!(body["reference_audio"], "UklGRg==");
        assert_eq!(body["reference_text"], "reference words");
    }

    #[test]
    fn request_body_sends_reference_audio_alone() {
        let dir = tempdir().expect("temp dir");
        let audio_path = dir.path().join("voice.wav");
        fs::write(&audio_path, b"RIFF").expect("audio fixture");

        let body = build_request_body(BuildRequestArgs {
            model: "Chatterbox-500M".to_string(),
            text: "hello".to_string(),
            speaker: None,
            saved_voice_id: None,
            reference_audio: Some(audio_path),
            reference_text: None,
            reference_text_file: None,
            instructions: None,
            format: &AudioFormat::Wav,
            speed: None,
            temperature: None,
            language: None,
            reference_language: None,
            controls: SpeechControls::default(),
            stream: false,
        })
        .expect("request body");

        assert_eq!(body["reference_audio"], "UklGRg==");
        assert!(body.get("reference_text").is_none());
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
            language: None,
            reference_language: None,
            controls: SpeechControls::default(),
            stream: false,
        })
        .expect_err("a transcript without audio should fail");

        assert!(err
            .to_string()
            .contains("--reference-text needs --reference-audio"));
    }
}
