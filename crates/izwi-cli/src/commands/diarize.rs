use crate::TranscriptFormat;
use crate::error::{CliError, Result};
use crate::http;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct DiarizeArgs {
    pub file: PathBuf,
    pub model: String,
    pub num_speakers: Option<u32>,
    pub format: TranscriptFormat,
    pub output: Option<PathBuf>,
    pub transcribe: bool,
    pub asr_model: String,
}

const DIARIZATION_POLL_TIMEOUT_SECS: u64 = 600;
const DIARIZATION_POLL_INTERVAL_MS: u64 = 500;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiarizationSegment {
    speaker: String,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiarizationWord {
    word: String,
    speaker: String,
    start: f32,
    end: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_confidence: Option<f32>,
    overlaps_segment: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiarizationUtterance {
    speaker: String,
    start: f32,
    end: f32,
    text: String,
    word_start: usize,
    word_end: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DiarizationJobRecord {
    id: String,
    processing_status: String,
    processing_error: Option<String>,
    #[serde(default)]
    segments: Vec<DiarizationSegment>,
    #[serde(default)]
    words: Vec<DiarizationWord>,
    #[serde(default)]
    utterances: Vec<DiarizationUtterance>,
    #[serde(default)]
    asr_text: String,
    #[serde(default)]
    raw_transcript: String,
    #[serde(default)]
    transcript: String,
    #[serde(default)]
    llm_refined: bool,
    alignment_coverage: Option<f32>,
    #[serde(default)]
    unattributed_words: usize,
    #[serde(default)]
    speaker_count: usize,
    duration_secs: Option<f32>,
    #[serde(default)]
    processing_time_ms: f64,
    rtf: Option<f64>,
}

#[derive(Debug, Serialize)]
struct JsonDiarizationOutput {
    segments: Vec<DiarizationSegment>,
    transcript: String,
}

#[derive(Debug, Serialize)]
struct VerboseJsonDiarizationOutput {
    segments: Vec<DiarizationSegment>,
    words: Vec<DiarizationWord>,
    utterances: Vec<DiarizationUtterance>,
    asr_text: String,
    raw_transcript: String,
    transcript: String,
    llm_refined: bool,
    alignment_coverage: f32,
    unattributed_words: usize,
    speaker_count: usize,
    duration: f32,
    processing_time_ms: f64,
    rtf: Option<f64>,
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

    if transcribe {
        eprintln!(
            "Note: --transcribe is now a compatibility flag; diarization responses already include transcript output."
        );
    }

    let mut request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
        "asr_model": asr_model,
    });
    if let Some(filename) = audio_filename_from_path(&file) {
        request_body["audio_filename"] = serde_json::Value::String(filename);
    }
    if let Some(mime_type) = audio_mime_type_from_path(&file) {
        request_body["audio_mime_type"] = serde_json::Value::String(mime_type.to_string());
    }

    if let Some(num) = num_speakers {
        request_body["min_speakers"] = serde_json::Value::Number(num.into());
        request_body["max_speakers"] = serde_json::Value::Number(num.into());
    }

    let client = http::client(Some(std::time::Duration::from_secs(600)))?;
    let response = client
        .post(create_diarization_job_url(server))
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

    let created = response
        .json::<DiarizationJobRecord>()
        .await
        .map_err(|e| CliError::Other(e.to_string()))?;
    let completed = wait_for_diarization_job(&client, server, created).await?;
    let result = format_diarization_output(&completed, format)?;

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

async fn wait_for_diarization_job(
    client: &reqwest::Client,
    server: &str,
    record: DiarizationJobRecord,
) -> Result<DiarizationJobRecord> {
    wait_for_diarization_job_with_polling(
        client,
        server,
        record,
        Duration::from_secs(DIARIZATION_POLL_TIMEOUT_SECS),
        Duration::from_millis(DIARIZATION_POLL_INTERVAL_MS),
    )
    .await
}

async fn wait_for_diarization_job_with_polling(
    client: &reqwest::Client,
    server: &str,
    mut record: DiarizationJobRecord,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<DiarizationJobRecord> {
    let started = Instant::now();
    loop {
        match record.processing_status.as_str() {
            "ready" => return Ok(record),
            "failed" => {
                return Err(CliError::ApiError {
                    status: 500,
                    message: record
                        .processing_error
                        .unwrap_or_else(|| "Diarization failed".to_string()),
                });
            }
            "pending" | "processing" => {}
            other => {
                return Err(CliError::Other(format!(
                    "Unexpected diarization status: {other}"
                )));
            }
        }

        if started.elapsed() >= timeout {
            return Err(CliError::Other(format!(
                "Timed out waiting for diarization job {}",
                record.id
            )));
        }

        tokio::time::sleep(poll_interval).await;
        let response = client
            .get(diarization_job_url(server, &record.id))
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

        record = response
            .json::<DiarizationJobRecord>()
            .await
            .map_err(|e| CliError::Other(e.to_string()))?;
    }
}

fn create_diarization_job_url(server: &str) -> String {
    format!(
        "{}/v1/speech-to-text/jobs?job_kind=diarization",
        server.trim_end_matches('/')
    )
}

fn diarization_job_url(server: &str, record_id: &str) -> String {
    format!(
        "{}/v1/speech-to-text/jobs/{}?job_kind=diarization",
        server.trim_end_matches('/'),
        record_id
    )
}

fn audio_filename_from_path(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains(['\r', '\n']))
        .map(ToOwned::to_owned)
}

fn audio_mime_type_from_path(path: &Path) -> Option<&'static str> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.trim().to_ascii_lowercase())?;
    match extension.as_str() {
        "wav" | "wave" => Some("audio/wav"),
        "mp3" => Some("audio/mpeg"),
        "m4a" | "mp4" => Some("audio/mp4"),
        "webm" => Some("audio/webm"),
        "ogg" | "oga" | "opus" => Some("audio/ogg"),
        "flac" => Some("audio/flac"),
        "aac" => Some("audio/aac"),
        _ => None,
    }
}

fn format_diarization_output(
    record: &DiarizationJobRecord,
    format: TranscriptFormat,
) -> Result<String> {
    match format {
        TranscriptFormat::Text => Ok(format_diarization_text(record)),
        TranscriptFormat::Json => serde_json::to_string(&JsonDiarizationOutput {
            segments: record.segments.clone(),
            transcript: record.transcript.clone(),
        })
        .map_err(CliError::Serialization),
        TranscriptFormat::VerboseJson => serde_json::to_string(&VerboseJsonDiarizationOutput {
            segments: record.segments.clone(),
            words: record.words.clone(),
            utterances: record.utterances.clone(),
            asr_text: record.asr_text.clone(),
            raw_transcript: record.raw_transcript.clone(),
            transcript: record.transcript.clone(),
            llm_refined: record.llm_refined,
            alignment_coverage: record.alignment_coverage.unwrap_or(0.0),
            unattributed_words: record.unattributed_words,
            speaker_count: record.speaker_count,
            duration: record.duration_secs.unwrap_or(0.0),
            processing_time_ms: record.processing_time_ms,
            rtf: record.rtf,
        })
        .map_err(CliError::Serialization),
    }
}

fn format_diarization_text(record: &DiarizationJobRecord) -> String {
    if !record.transcript.trim().is_empty() {
        return record.transcript.clone();
    }

    let mut out = String::new();
    for segment in &record.segments {
        let duration = (segment.end - segment.start).max(0.0);
        out.push_str(&format!(
            "SPEAKER unknown 1 {:.3} {:.3} <NA> <NA> {} <NA> <NA>\n",
            segment.start, duration, segment.speaker
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    #[derive(Debug)]
    struct CapturedRequest {
        request_line: String,
        body: String,
    }

    #[tokio::test]
    async fn execute_posts_to_canonical_job_route_and_polls_until_ready() {
        let temp_dir = tempfile::tempdir().expect("temp dir");
        let audio_path = temp_dir.path().join("sample.wav");
        let output_path = temp_dir.path().join("diarization.json");
        tokio::fs::write(&audio_path, b"fake wav")
            .await
            .expect("write audio");

        let (server, server_task) = spawn_flow_server(vec![
            http_json_response("202 Accepted", &sample_record("job-1", "pending")),
            http_json_response("200 OK", &ready_record("job-1")),
        ])
        .await;

        execute(
            DiarizeArgs {
                file: audio_path,
                model: "nvidia/sortformer".to_string(),
                num_speakers: Some(2),
                format: TranscriptFormat::Json,
                output: Some(output_path.clone()),
                transcribe: false,
                asr_model: "distil-whisper".to_string(),
            },
            &format!("{server}/"),
        )
        .await
        .expect("diarization command should complete");

        let requests = server_task.await.expect("server task");
        assert_eq!(
            requests[0].request_line,
            "POST /v1/speech-to-text/jobs?job_kind=diarization HTTP/1.1"
        );
        let request_json: Value =
            serde_json::from_str(&requests[0].body).expect("valid request json");
        assert_eq!(request_json["model"], "nvidia/sortformer");
        assert_eq!(request_json["asr_model"], "distil-whisper");
        assert_eq!(request_json["audio_filename"], "sample.wav");
        assert_eq!(request_json["audio_mime_type"], "audio/wav");
        assert_eq!(request_json["min_speakers"], 2);
        assert_eq!(request_json["max_speakers"], 2);
        assert_eq!(
            request_json["audio_base64"],
            Value::String(STANDARD.encode(b"fake wav"))
        );
        assert_eq!(
            requests[1].request_line,
            "GET /v1/speech-to-text/jobs/job-1?job_kind=diarization HTTP/1.1"
        );

        let output = tokio::fs::read_to_string(output_path)
            .await
            .expect("read output");
        let output_json: Value = serde_json::from_str(&output).expect("valid output json");
        assert_eq!(output_json["transcript"], "Speaker one text");
        assert_eq!(output_json["segments"][0]["speaker"], "SPEAKER_00");
    }

    #[tokio::test]
    async fn wait_for_diarization_job_maps_failed_records_to_api_error() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client");
        let err = wait_for_diarization_job(
            &client,
            "http://127.0.0.1:1",
            failed_record("job-2", "model failed"),
        )
        .await
        .expect_err("failed records should return api error");

        match err {
            CliError::ApiError { status, message } => {
                assert_eq!(status, 500);
                assert_eq!(message, "model failed");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[tokio::test]
    async fn wait_for_diarization_job_times_out_pending_records() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("client");
        let err = wait_for_diarization_job_with_polling(
            &client,
            "http://127.0.0.1:1",
            sample_record("job-timeout", "pending"),
            Duration::ZERO,
            Duration::ZERO,
        )
        .await
        .expect_err("pending records should time out");

        assert!(
            matches!(err, CliError::Other(message) if message == "Timed out waiting for diarization job job-timeout")
        );
    }

    #[test]
    fn diarization_job_urls_trim_trailing_server_slash() {
        assert_eq!(
            create_diarization_job_url("http://localhost:8080/"),
            "http://localhost:8080/v1/speech-to-text/jobs?job_kind=diarization"
        );
        assert_eq!(
            diarization_job_url("http://localhost:8080/", "abc"),
            "http://localhost:8080/v1/speech-to-text/jobs/abc?job_kind=diarization"
        );
    }

    #[test]
    fn audio_upload_metadata_uses_basename_and_known_mime() {
        let path = PathBuf::from("/Users/example/private/call.OPUS");

        assert_eq!(
            audio_filename_from_path(&path).as_deref(),
            Some("call.OPUS")
        );
        assert_eq!(audio_mime_type_from_path(&path), Some("audio/ogg"));
        assert_eq!(audio_mime_type_from_path(Path::new("unknown.bin")), None);
    }

    async fn spawn_flow_server(
        responses: Vec<String>,
    ) -> (String, tokio::task::JoinHandle<Vec<CapturedRequest>>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind test server");
        let addr = listener.local_addr().expect("server addr");
        let handle = tokio::spawn(async move {
            let mut requests = Vec::new();
            for response in responses {
                let (mut stream, _) = listener.accept().await.expect("accept request");
                requests.push(read_request(&mut stream).await);
                stream
                    .write_all(response.as_bytes())
                    .await
                    .expect("write response");
            }
            requests
        });
        (format!("http://{addr}"), handle)
    }

    async fn read_request(stream: &mut TcpStream) -> CapturedRequest {
        let mut bytes = Vec::new();
        let mut buffer = [0_u8; 1024];

        let header_end = loop {
            let read = stream.read(&mut buffer).await.expect("read request");
            assert!(read > 0, "connection closed before headers");
            bytes.extend_from_slice(&buffer[..read]);
            if let Some(index) = find_header_end(&bytes) {
                break index;
            }
        };

        let headers = String::from_utf8_lossy(&bytes[..header_end]).to_string();
        let content_length = headers
            .lines()
            .find_map(|line| {
                line.strip_prefix("content-length: ")
                    .or_else(|| line.strip_prefix("Content-Length: "))
                    .and_then(|value| value.parse::<usize>().ok())
            })
            .unwrap_or(0);
        let body_start = header_end + 4;
        let total_len = body_start + content_length;
        while bytes.len() < total_len {
            let read = stream.read(&mut buffer).await.expect("read body");
            assert!(read > 0, "connection closed before body");
            bytes.extend_from_slice(&buffer[..read]);
        }

        CapturedRequest {
            request_line: headers.lines().next().unwrap_or_default().to_string(),
            body: String::from_utf8_lossy(&bytes[body_start..total_len]).to_string(),
        }
    }

    fn find_header_end(bytes: &[u8]) -> Option<usize> {
        bytes.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn http_json_response(status: &str, record: &DiarizationJobRecord) -> String {
        let body = serde_json::to_string(record).expect("serialize record");
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn sample_record(id: &str, status: &str) -> DiarizationJobRecord {
        DiarizationJobRecord {
            id: id.to_string(),
            processing_status: status.to_string(),
            processing_error: None,
            segments: Vec::new(),
            words: Vec::new(),
            utterances: Vec::new(),
            asr_text: String::new(),
            raw_transcript: String::new(),
            transcript: String::new(),
            llm_refined: false,
            alignment_coverage: None,
            unattributed_words: 0,
            speaker_count: 0,
            duration_secs: None,
            processing_time_ms: 0.0,
            rtf: None,
        }
    }

    fn ready_record(id: &str) -> DiarizationJobRecord {
        DiarizationJobRecord {
            processing_status: "ready".to_string(),
            segments: vec![DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start: 0.0,
                end: 1.25,
                confidence: Some(0.95),
            }],
            words: vec![DiarizationWord {
                word: "Speaker".to_string(),
                speaker: "SPEAKER_00".to_string(),
                start: 0.0,
                end: 0.4,
                speaker_confidence: Some(0.93),
                overlaps_segment: true,
            }],
            utterances: vec![DiarizationUtterance {
                speaker: "SPEAKER_00".to_string(),
                start: 0.0,
                end: 1.25,
                text: "Speaker one text".to_string(),
                word_start: 0,
                word_end: 3,
            }],
            transcript: "Speaker one text".to_string(),
            asr_text: "Speaker one text".to_string(),
            raw_transcript: "speaker one text".to_string(),
            llm_refined: true,
            alignment_coverage: Some(1.0),
            unattributed_words: 0,
            speaker_count: 1,
            duration_secs: Some(1.25),
            processing_time_ms: 12.5,
            rtf: Some(0.1),
            ..sample_record(id, "ready")
        }
    }

    fn failed_record(id: &str, message: &str) -> DiarizationJobRecord {
        DiarizationJobRecord {
            processing_status: "failed".to_string(),
            processing_error: Some(message.to_string()),
            ..sample_record(id, "failed")
        }
    }
}
