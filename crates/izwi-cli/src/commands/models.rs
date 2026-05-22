use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::utils;
use crate::{ModelCommands, OutputFormat};
use comfy_table::{Cell, CellAlignment, Color, Table};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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
    quiet: bool,
) -> Result<()> {
    match command {
        ModelCommands::List { local, detailed } => {
            list_models(server, local, detailed, format, quiet).await
        }
        ModelCommands::Info { model, json } => show_model_info(server, &model, json, format).await,
        ModelCommands::Load { model, wait } => load_model(server, &model, wait).await,
        ModelCommands::Unload { model, yes } => unload_model(server, &model, yes).await,
        ModelCommands::Progress { model } => show_download_progress(server, model.as_deref()).await,
    }
}

async fn list_models(
    server: &str,
    local: bool,
    detailed: bool,
    format: OutputFormat,
    quiet: bool,
) -> Result<()> {
    if !quiet {
        println!("{}", style("Fetching models...").dim());
    }

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models", server))
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

    let payload: AdminModelsResponse = response.json().await?;
    let mut models = payload.models;
    for model in &mut models {
        reconcile_local_state(model);
    }

    if local {
        models.retain(|m| m.status != "not_downloaded");
    }

    match format {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&models)?);
        }
        OutputFormat::Yaml => {
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
    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if response.status().as_u16() == 404 {
        return Err(CliError::ModelNotFound(model.to_string()));
    }

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let info: AdminModelInfo = response.json().await?;
    let mut info = info;
    reconcile_local_state(&mut info);

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

async fn load_model(server: &str, model: &str, wait: bool) -> Result<()> {
    let theme = Theme::default();
    theme.step(1, 3, "Loading model...");

    let client = http::client(Some(std::time::Duration::from_secs(60)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/load", server, model))
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

    theme.success(&format!("Load requested for '{}'", model));

    if wait {
        theme.step(2, 3, "Waiting for model readiness...");
        for _ in 0..60 {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

            let resp = client
                .get(format!("{}/v1/admin/models/{}", server, model))
                .send()
                .await
                .map_err(|e| CliError::ConnectionError(e.to_string()))?;

            if !resp.status().is_success() {
                continue;
            }

            let info: AdminModelInfo = resp.json().await?;
            let mut info = info;
            reconcile_local_state(&mut info);
            if info.status == "ready" {
                theme.step(3, 3, "Model is ready");
                return Ok(());
            }
            if info.status == "error" {
                return Err(CliError::Other(
                    info.error_message
                        .unwrap_or_else(|| "Model entered error state".to_string()),
                ));
            }
        }

        return Err(CliError::Other(
            "Timed out waiting for model to become ready".to_string(),
        ));
    }

    theme.step(3, 3, "Done");
    Ok(())
}

async fn unload_model(server: &str, model: &str, yes: bool) -> Result<()> {
    let theme = Theme::default();

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

    theme.success(&format!("Model '{}' unloaded successfully", model));
    Ok(())
}

async fn show_download_progress(server: &str, model: Option<&str>) -> Result<()> {
    let Some(model) = model else {
        println!("Please provide a model variant: izwi models progress <model>");
        return Ok(());
    };

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .get(format!("{}/v1/admin/models/{}", server, model))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if response.status().as_u16() == 404 {
        return Err(CliError::ModelNotFound(model.to_string()));
    }
    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let info: AdminModelInfo = response.json().await?;
    let mut info = info;
    reconcile_local_state(&mut info);
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

fn reconcile_local_state(model: &mut AdminModelInfo) {
    if model.status != "not_downloaded" || model.local_path.is_some() {
        return;
    }

    if let Some(path) = utils::model_dir_if_present(&model.variant) {
        model.status = "downloaded".to_string();
        if model.size_bytes.is_none() {
            model.size_bytes = utils::directory_size_bytes(&path);
        }
        model.local_path = Some(path);
        if model.download_progress.is_none() {
            model.download_progress = Some(100.0);
        }
    }
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
}
