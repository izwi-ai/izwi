use crate::error::{CliError, Result};
use crate::http;
use crate::style::{self, Theme};
use crate::{ModelCommands, OutputFormat};
use comfy_table::{Cell, CellAlignment, Color, Table};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const LOAD_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// `POST /v1/admin/models/{variant}/load`'s answer.
#[derive(Debug, Deserialize)]
struct LoadResponse {
    status: String,
    #[serde(default)]
    warning: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AdminModelsResponse {
    models: Vec<AdminModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdminModelInfo {
    variant: String,
    #[serde(default)]
    enabled: Option<bool>,
    status: String,
    local_path: Option<PathBuf>,
    size_bytes: Option<u64>,
    download_progress: Option<f32>,
    error_message: Option<String>,
    #[serde(default)]
    modalities: Vec<String>,
    #[serde(default)]
    route_capabilities: AdminModelRouteCapabilities,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AdminModelRouteCapabilities {
    #[serde(default)]
    openai_chat_completions: bool,
    #[serde(default)]
    openai_responses: bool,
    #[serde(default)]
    openai_audio_speech: bool,
    #[serde(default)]
    openai_audio_transcriptions: bool,
    #[serde(default)]
    speech_to_text_jobs: bool,
    #[serde(default)]
    speech_to_text_realtime: bool,
    #[serde(default)]
    diarization_records: bool,
    #[serde(default)]
    text_to_speech_records: bool,
    #[serde(default)]
    voice_design_records: bool,
    #[serde(default)]
    voice_clone_records: bool,
    #[serde(default)]
    saved_voice_reuse: bool,
    #[serde(default)]
    studio_projects: bool,
    #[serde(default)]
    voice_realtime_text_model: bool,
    #[serde(default)]
    voice_realtime_modular_asr: bool,
    #[serde(default)]
    voice_realtime_modular_tts: bool,
    #[serde(default)]
    voice_realtime_unified: bool,
    #[serde(default)]
    forced_alignment: bool,
    #[serde(default)]
    tokenizer: bool,
}

pub async fn execute(
    command: ModelCommands,
    server: &str,
    format: OutputFormat,
    theme: &Theme,
) -> Result<()> {
    match command {
        ModelCommands::List { local, detailed } => {
            list_models(server, local, detailed, format).await
        }
        ModelCommands::Info { model, json } => show_model_info(server, &model, json, format).await,
        // Loading always waits; `--wait` is accepted for older scripts.
        ModelCommands::Load { model, timeout, .. } => {
            load_model(server, &model, Duration::from_secs(timeout), theme).await
        }
        ModelCommands::Unload { model, yes } => unload_model(server, &model, yes, theme).await,
        ModelCommands::Progress { model } => show_download_progress(server, &model).await,
    }
}

async fn list_models(
    server: &str,
    local: bool,
    detailed: bool,
    format: OutputFormat,
) -> Result<()> {
    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models", server))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }

    let payload: AdminModelsResponse = response.json().await?;
    let mut models = payload.models;

    if local {
        models.retain(|m| m.status != "not_downloaded");
    }

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        OutputFormat::Plain => {
            for model in &models {
                println!("{}", model.variant);
            }
        }
        OutputFormat::Table => {
            print_models_table(&models, detailed);
        }
    }

    Ok(())
}

fn print_models_table(models: &[AdminModelInfo], detailed: bool) {
    let mut table = Table::new();
    if detailed {
        table.set_header(vec!["Variant", "Status", "Size", "Progress", "Local Path"]);
    } else {
        table.set_header(vec!["Variant", "Status", "Size"]);
    }

    for model in models {
        let size = model
            .size_bytes
            .map(|s| humansize::format_size(s, humansize::BINARY))
            .unwrap_or_else(|| "-".to_string());

        if detailed {
            let progress = model
                .download_progress
                .map(|p| format!("{:.1}%", p))
                .unwrap_or_else(|| "-".to_string());
            let local_path = model
                .local_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string());
            table.add_row(vec![
                Cell::new(&model.variant).fg(Color::Cyan),
                status_cell(&model.status),
                Cell::new(size).set_alignment(CellAlignment::Right),
                Cell::new(progress).set_alignment(CellAlignment::Right),
                Cell::new(local_path),
            ]);
        } else {
            table.add_row(vec![
                Cell::new(&model.variant).fg(Color::Cyan),
                status_cell(&model.status),
                Cell::new(size).set_alignment(CellAlignment::Right),
            ]);
        }
    }

    println!("{}", table);
}

fn status_cell(status: &str) -> Cell {
    let color = match status {
        "ready" | "downloaded" => Color::Green,
        "downloading" => Color::Yellow,
        "loading" => Color::Blue,
        "not_downloaded" => Color::DarkGrey,
        "error" => Color::Red,
        _ => Color::DarkGrey,
    };
    Cell::new(status).fg(color)
}

fn status_color(status: &str) -> String {
    match status {
        "ready" => style(status).green().to_string(),
        "downloaded" => style(status).green().to_string(),
        "downloading" => style(status).yellow().to_string(),
        "loading" => style(status).blue().to_string(),
        "not_downloaded" => style(status).dim().to_string(),
        "error" => style(status).red().to_string(),
        _ => style(status).dim().to_string(),
    }
}

fn route_capability_labels(capabilities: &AdminModelRouteCapabilities) -> Vec<&'static str> {
    let mut labels = Vec::new();
    if capabilities.openai_chat_completions {
        labels.push("chat completions");
    }
    if capabilities.openai_responses {
        labels.push("responses");
    }
    if capabilities.openai_audio_speech {
        labels.push("audio speech");
    }
    if capabilities.openai_audio_transcriptions {
        labels.push("audio transcriptions");
    }
    if capabilities.speech_to_text_jobs {
        labels.push("speech-to-text jobs");
    }
    if capabilities.speech_to_text_realtime {
        labels.push("speech-to-text realtime");
    }
    if capabilities.diarization_records {
        labels.push("diarization records");
    }
    if capabilities.text_to_speech_records {
        labels.push("text-to-speech records");
    }
    if capabilities.voice_design_records {
        labels.push("voice design");
    }
    if capabilities.voice_clone_records {
        labels.push("voice clone");
    }
    if capabilities.saved_voice_reuse {
        labels.push("saved voice reuse");
    }
    if capabilities.studio_projects {
        labels.push("studio projects");
    }
    if capabilities.voice_realtime_text_model {
        labels.push("voice realtime text model");
    }
    if capabilities.voice_realtime_modular_asr {
        labels.push("voice realtime ASR");
    }
    if capabilities.voice_realtime_modular_tts {
        labels.push("voice realtime TTS");
    }
    if capabilities.voice_realtime_unified {
        labels.push("voice realtime unified");
    }
    if capabilities.forced_alignment {
        labels.push("forced alignment");
    }
    if capabilities.tokenizer {
        labels.push("tokenizer");
    }
    labels
}

async fn show_model_info(
    server: &str,
    model: &str,
    json: bool,
    format: OutputFormat,
) -> Result<()> {
    let info = fetch_model_info(server, model).await?;

    if json || matches!(format, OutputFormat::Json) {
        println!("{}", serde_json::to_string_pretty(&info)?);
    } else {
        println!("{}: {}", style("Variant").bold(), info.variant);
        println!("{}: {}", style("Status").bold(), status_color(&info.status));
        println!(
            "{}: {}",
            style("Size").bold(),
            info.size_bytes
                .map(|s| humansize::format_size(s, humansize::BINARY))
                .unwrap_or_else(|| "-".to_string())
        );
        println!(
            "{}: {}",
            style("Progress").bold(),
            info.download_progress
                .map(|p| format!("{:.1}%", p))
                .unwrap_or_else(|| "-".to_string())
        );
        println!(
            "{}: {}",
            style("Local Path").bold(),
            info.local_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string())
        );
        if !info.modalities.is_empty() {
            println!(
                "{}: {}",
                style("Modalities").bold(),
                info.modalities.join(", ")
            );
        }
        let route_labels = route_capability_labels(&info.route_capabilities);
        if !route_labels.is_empty() {
            println!("{}: {}", style("Routes").bold(), route_labels.join(", "));
        }
        if let Some(err) = info.error_message {
            println!("{}: {}", style("Error").bold(), style(err).red());
        }
    }

    Ok(())
}

/// The server's record of `model`; a model it does not know is
/// [`CliError::ModelNotFound`].
async fn fetch_model_info(server: &str, model: &str) -> Result<AdminModelInfo> {
    let client = http::client(Some(Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(CliError::ModelNotFound(model.to_string()));
    }
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    Ok(response.json().await?)
}

async fn load_model(server: &str, model: &str, timeout: Duration, theme: &Theme) -> Result<()> {
    let info = fetch_model_info(server, model).await?;
    match info.status.as_str() {
        "ready" => {
            theme.success(&format!("'{}' is already loaded", info.variant));
            return Ok(());
        }
        "not_downloaded" => {
            return Err(CliError::InvalidInput(format!(
                "'{0}' is not downloaded. Download it first: izwi pull {0}",
                info.variant
            )))
        }
        "downloading" => {
            return Err(CliError::InvalidInput(format!(
                "'{0}' is still downloading ({1:.0}%). Load it once `izwi models progress {0}` shows it downloaded.",
                info.variant,
                info.download_progress.unwrap_or(0.0)
            )))
        }
        _ => {}
    }

    let progress = style::spinner(format!("Loading '{}'...", info.variant));
    let loaded = wait_for_load(server, &info.variant, timeout).await;
    progress.finish_and_clear();
    if let Some(warning) = loaded? {
        theme.warning(&warning);
    }
    theme.success(&format!("'{}' is loaded and ready", info.variant));
    Ok(())
}

/// Asks the server to load `model` and waits until it is ready, or fails
/// once `timeout` has passed. Returns the server's load warning, if any.
async fn wait_for_load(server: &str, model: &str, timeout: Duration) -> Result<Option<String>> {
    let deadline = Instant::now() + timeout;
    let timed_out = || {
        CliError::Other(format!(
            "'{model}' did not finish loading within {}s. The server may still be loading it: \
             check with `izwi models info {model}`, or allow longer with --timeout.",
            timeout.as_secs()
        ))
    };

    let client = http::client(Some(timeout))?;
    let response = match client
        .post(format!("{}/v1/admin/models/{}/load", server, model))
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) if error.is_timeout() => return Err(timed_out()),
        Err(error) => return Err(error.into()),
    };
    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }
    let answer: LoadResponse = response.json().await?;
    if answer.status == "loaded" {
        return Ok(answer.warning);
    }

    // The server answered before the model was ready: follow its state.
    loop {
        if Instant::now() >= deadline {
            return Err(timed_out());
        }
        tokio::time::sleep(LOAD_POLL_INTERVAL).await;
        let info = fetch_model_info(server, model).await?;
        match info.status.as_str() {
            "ready" => return Ok(answer.warning),
            "error" => {
                return Err(CliError::Other(
                    info.error_message
                        .unwrap_or_else(|| format!("'{model}' failed to load")),
                ))
            }
            _ => {}
        }
    }
}

async fn unload_model(server: &str, model: &str, yes: bool, theme: &Theme) -> Result<()> {
    if !yes {
        println!(
            "This will unload model '{}' from memory.",
            style(model).cyan()
        );
        let confirm = dialoguer::Confirm::new()
            .with_prompt("Are you sure?")
            .default(false)
            .interact()
            .map_err(|e| CliError::Other(e.to_string()))?;

        if !confirm {
            println!("Cancelled.");
            return Ok(());
        }
    }

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/unload", server, model))
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(CliError::from_response(response).await);
    }

    theme.success(&format!("Model '{}' unloaded successfully", model));
    Ok(())
}

async fn show_download_progress(server: &str, model: &str) -> Result<()> {
    let info = fetch_model_info(server, model).await?;
    println!("{}: {}", style("Variant").bold(), info.variant);
    println!("{}: {}", style("Status").bold(), status_color(&info.status));
    println!(
        "{}: {}",
        style("Progress").bold(),
        info.download_progress
            .map(|p| format!("{:.1}%", p))
            .unwrap_or_else(|| "-".to_string())
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_capability_labels_show_voxtral_offline_transcription_only() {
        let capabilities = AdminModelRouteCapabilities {
            openai_audio_transcriptions: true,
            speech_to_text_jobs: true,
            speech_to_text_realtime: false,
            voice_realtime_modular_asr: false,
            voice_realtime_unified: false,
            ..Default::default()
        };

        let labels = route_capability_labels(&capabilities);

        assert_eq!(labels, vec!["audio transcriptions", "speech-to-text jobs"]);
    }

    #[test]
    fn admin_model_info_accepts_route_capabilities_from_server() {
        let raw = serde_json::json!({
            "variant": "Voxtral-Mini-4B-Realtime-2602",
            "enabled": true,
            "status": "downloaded",
            "local_path": null,
            "size_bytes": null,
            "download_progress": null,
            "error_message": null,
            "modalities": ["audio_input", "text_output"],
            "route_capabilities": {
                "openai_audio_transcriptions": true,
                "speech_to_text_jobs": true,
                "speech_to_text_realtime": false,
                "voice_realtime_unified": false
            }
        });

        let info: AdminModelInfo = serde_json::from_value(raw).unwrap();

        assert_eq!(info.variant, "Voxtral-Mini-4B-Realtime-2602");
        assert_eq!(info.enabled, Some(true));
        assert_eq!(info.modalities, vec!["audio_input", "text_output"]);
        assert!(info.route_capabilities.openai_audio_transcriptions);
        assert!(info.route_capabilities.speech_to_text_jobs);
        assert!(!info.route_capabilities.speech_to_text_realtime);
    }

    /// A server that answers each request in turn with `responses`, or
    /// holds the connection open when they run out.
    async fn scripted_server(responses: Vec<String>) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let mut responses = responses.into_iter();
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0u8; 4096];
                let _ = stream.read(&mut request).await;
                match responses.next() {
                    Some(response) => stream.write_all(response.as_bytes()).await.unwrap(),
                    None => {
                        tokio::time::sleep(Duration::from_secs(30)).await;
                        return;
                    }
                }
            }
        });
        server
    }

    fn json_response(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[tokio::test]
    async fn a_load_that_outlasts_the_timeout_says_so() {
        let server = scripted_server(Vec::new()).await;
        let error = wait_for_load(&server, "Qwen3.5-0.8B", Duration::from_millis(300))
            .await
            .expect_err("the load never answers");
        let message = error.to_string();
        assert!(message.contains("did not finish loading"), "{message}");
        assert!(message.contains("--timeout"), "{message}");
    }

    #[tokio::test]
    async fn a_load_answered_early_follows_the_server_state_until_ready() {
        let info = |status: &str| {
            json_response(&format!(
                r#"{{"variant":"Qwen3.5-0.8B","status":"{status}","local_path":null,"size_bytes":null,"download_progress":null,"error_message":null}}"#
            ))
        };
        let server = scripted_server(vec![
            json_response(r#"{"status":"loading","message":"started"}"#),
            info("loading"),
            info("ready"),
        ])
        .await;
        let warning = wait_for_load(&server, "Qwen3.5-0.8B", Duration::from_secs(20))
            .await
            .expect("the model becomes ready");
        assert_eq!(warning, None);

        let server = scripted_server(vec![json_response(
            r#"{"status":"loaded","message":"ok","warning":"low memory"}"#,
        )])
        .await;
        let warning = wait_for_load(&server, "Qwen3.5-0.8B", Duration::from_secs(5))
            .await
            .expect("a synchronous load");
        assert_eq!(warning.as_deref(), Some("low memory"));
    }
}
