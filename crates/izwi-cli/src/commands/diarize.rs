use crate::TranscriptFormat;
use crate::error::{CliError, Result};
use crate::http;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
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

#[derive(Debug, Deserialize)]
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

    if let Some(num) = num_speakers {
        request_body["min_speakers"] = serde_json::Value::Number(num.into());
        request_body["max_speakers"] = serde_json::Value::Number(num.into());
    }

    let client = http::client(Some(std::time::Duration::from_secs(600)))?;
    let response = client
        .post(format!(
            "{}/v1/speech-to-text/jobs?job_kind=diarization",
            server
        ))
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
    mut record: DiarizationJobRecord,
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

        if started.elapsed() >= Duration::from_secs(DIARIZATION_POLL_TIMEOUT_SECS) {
            return Err(CliError::Other(format!(
                "Timed out waiting for diarization job {}",
                record.id
            )));
        }

        tokio::time::sleep(Duration::from_millis(DIARIZATION_POLL_INTERVAL_MS)).await;
        let response = client
            .get(format!(
                "{}/v1/speech-to-text/jobs/{}?job_kind=diarization",
                server, record.id
            ))
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
