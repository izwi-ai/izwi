use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::utils;
use console::style;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use izwi_core::parse_model_variant;
use serde::Deserialize;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct ModelState {
    variant: String,
    status: String,
    local_path: Option<PathBuf>,
    download_progress: Option<f32>,
    error_message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DownloadResponse {
    status: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct ProgressEvent {
    percent: f32,
    status: String,
    current_file: String,
    files_completed: usize,
    files_total: usize,
}

const DOWNLOAD_POLL_INTERVAL: Duration = Duration::from_secs(1);
const DOWNLOAD_WAIT_TIMEOUT: Duration = Duration::from_secs(60 * 60);

pub async fn execute(
    model: String,
    force: bool,
    yes: bool,
    server: &str,
    theme: &Theme,
) -> Result<()> {
    // Check if model already exists
    if !force {
        let client = http::client(Some(std::time::Duration::from_secs(30)))?;
        let resp = client
            .get(format!("{}/v1/admin/models/{}", server, model))
            .send()
            .await;

        if let Ok(r) = resp {
            if r.status().as_u16() == 200 {
                if let Ok(state) = r.json::<ModelState>().await {
                    let has_local_files = state
                        .local_path
                        .as_ref()
                        .filter(|path| path.is_dir())
                        .is_some()
                        || utils::model_dir_if_present(&state.variant).is_some();

                    if state.status != "not_downloaded" || has_local_files {
                        theme.info(&format!(
                            "Model '{}' is currently '{}'",
                            model, state.status
                        ));
                        if !yes {
                            let confirm = dialoguer::Confirm::new()
                                .with_prompt("Re-download?")
                                .default(false)
                                .interact()
                                .map_err(|e| CliError::Other(e.to_string()))?;

                            if !confirm {
                                println!("Cancelled.");
                                return Ok(());
                            }
                        }
                    }
                }
            }
        }
    }

    theme.step(1, 3, &format!("Starting download for '{}'...", model));

    let client = http::client(Some(std::time::Duration::from_secs(30)))?;
    let response = client
        .post(format!("{}/v1/admin/models/{}/download", server, model))
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

    if let Ok(download_response) = response.json::<DownloadResponse>().await {
        match download_response.status.as_str() {
            "started" => theme.info(&download_response.message),
            "downloading" => theme.info("Download already in progress. Waiting for completion."),
            _ => {}
        }
    }

    theme.step(2, 3, "Downloading model files...");
    wait_for_download_completion(server, &model).await?;

    theme.step(3, 3, "Finalizing...");

    theme.success(&format!("Model '{}' downloaded successfully!", model));
    println!();
    println!("Next steps:");
    println!(
        "  - Load the model: {}",
        style(format!("izwi models load {}", model)).cyan()
    );
    println!(
        "  - {}: {}",
        post_download_usage_label(&model),
        style(post_download_usage_command(&model)).cyan()
    );

    Ok(())
}

fn post_download_usage_label(model: &str) -> &'static str {
    match parse_model_variant(model).ok() {
        Some(variant) if variant.is_asr() || variant.is_voxtral() => "Transcribe audio",
        Some(variant) if variant.is_diarization() => "Diarize audio",
        Some(variant) if variant.is_chat() => "Start chat",
        Some(variant) if variant.is_tokenizer() => "Inspect tokenizer",
        _ => "Generate speech",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn post_download_hint_for_voxtral_uses_transcribe() {
        let model = "Voxtral-Mini-4B-Realtime-2602";

        assert_eq!(post_download_usage_label(model), "Transcribe audio");
        assert_eq!(
            post_download_usage_command(model),
            "izwi transcribe audio.wav --model Voxtral-Mini-4B-Realtime-2602"
        );
    }

    #[test]
    fn post_download_hint_for_tts_still_uses_tts() {
        let model = "Kokoro-82M";

        assert_eq!(post_download_usage_label(model), "Generate speech");
        assert_eq!(
            post_download_usage_command(model),
            "izwi tts 'Hello' -m Kokoro-82M"
        );
    }
}

fn post_download_usage_command(model: &str) -> String {
    match parse_model_variant(model).ok() {
        Some(variant) if variant.is_asr() || variant.is_voxtral() => {
            format!("izwi transcribe audio.wav --model {}", model)
        }
        Some(variant) if variant.is_diarization() => {
            format!("izwi diarize audio.wav --model {}", model)
        }
        Some(variant) if variant.is_chat() => format!("izwi chat --model {}", model),
        Some(variant) if variant.is_tokenizer() => format!("izwi models info {} --json", model),
        _ => format!("izwi tts 'Hello' -m {}", model),
    }
}

async fn wait_for_download_completion(server: &str, model: &str) -> Result<()> {
    if wait_for_download_stream(server, model).await? {
        return Ok(());
    }

    wait_for_download_poll(server, model).await
}

async fn wait_for_download_stream(server: &str, model: &str) -> Result<bool> {
    let client = http::client(None)?;
    let response = client
        .get(format!(
            "{}/v1/admin/models/{}/download/progress",
            server, model
        ))
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        return Ok(false);
    }

    let pb = ProgressBar::new(1000);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent}% {msg}")
            .map_err(|e| CliError::Other(e.to_string()))?
            .progress_chars("#>-"),
    );
    pb.set_message("Waiting for progress events...");

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(bytes) => bytes,
            Err(_) => {
                pb.abandon_with_message("Live stream interrupted");
                return Ok(false);
            }
        };

        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(newline) = buffer.find('\n') {
            let mut line = buffer[..newline].to_string();
            buffer.drain(..=newline);

            if line.ends_with('\r') {
                line.pop();
            }

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };

            let data = data.trim();
            if data.is_empty() {
                continue;
            }

            let event = match serde_json::from_str::<ProgressEvent>(data) {
                Ok(event) => event,
                Err(_) => continue,
            };

            let percent = event.percent.clamp(0.0, 100.0);
            pb.set_position((percent * 10.0).round() as u64);
            pb.set_message(format!(
                "{:.1}% {} ({}/{})",
                percent, event.current_file, event.files_completed, event.files_total
            ));

            if event.status == "completed" {
                pb.finish_with_message("Download complete");
                return Ok(true);
            }
        }
    }

    pb.finish_and_clear();
    Ok(false)
}

async fn wait_for_download_poll(server: &str, model: &str) -> Result<()> {
    let client = http::client(Some(Duration::from_secs(30)))?;
    let started = std::time::Instant::now();
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} [{elapsed_precise}] {msg}")
            .map_err(|e| CliError::Other(e.to_string()))?,
    );
    pb.enable_steady_tick(Duration::from_millis(120));

    let mut seen_active_status = false;

    loop {
        if started.elapsed() >= DOWNLOAD_WAIT_TIMEOUT {
            pb.abandon_with_message("Download timed out");
            return Err(CliError::Other(format!(
                "Timed out waiting for '{}' to finish downloading after {} seconds",
                model,
                DOWNLOAD_WAIT_TIMEOUT.as_secs()
            )));
        }

        let response = client
            .get(format!("{}/v1/admin/models/{}", server, model))
            .send()
            .await;

        let response = match response {
            Ok(response) => response,
            Err(err) => {
                pb.set_message(format!("Waiting for server status... ({})", err));
                tokio::time::sleep(DOWNLOAD_POLL_INTERVAL).await;
                continue;
            }
        };

        if response.status().as_u16() == 404 {
            pb.abandon_with_message("Model not found");
            return Err(CliError::ModelNotFound(model.to_string()));
        }

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let text = response.text().await.unwrap_or_default();
            pb.abandon_with_message("Download failed");
            return Err(CliError::ApiError {
                status,
                message: text,
            });
        }

        let state: ModelState = response.json().await?;

        if matches!(state.status.as_str(), "downloading" | "loading") {
            seen_active_status = true;
        }

        if matches!(state.status.as_str(), "downloaded" | "ready") {
            pb.finish_with_message("Download complete");
            return Ok(());
        }

        if state.status == "error" {
            pb.abandon_with_message("Download failed");
            return Err(CliError::Other(
                state
                    .error_message
                    .unwrap_or_else(|| "Model download failed".to_string()),
            ));
        }

        if seen_active_status && state.status == "not_downloaded" {
            pb.abandon_with_message("Download interrupted");
            return Err(CliError::Other(format!(
                "Download for '{}' was interrupted before completion",
                model
            )));
        }

        let progress = state
            .download_progress
            .map(|p| format!("{:.1}%", p))
            .unwrap_or_else(|| "-".to_string());
        pb.set_message(format!("status={} progress={}", state.status, progress));
        tokio::time::sleep(DOWNLOAD_POLL_INTERVAL).await;
    }
}
