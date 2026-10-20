use crate::error::{CliError, Result};
use crate::http;
use crate::OutputFormat;
use comfy_table::{Cell, CellAlignment, Color, Table};
use console::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize, Serialize)]
struct AdminModelsResponse {
    models: Vec<ModelRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ModelRecord {
    variant: String,
    status: String,
    local_path: Option<PathBuf>,
    size_bytes: Option<u64>,
    download_progress: Option<f32>,
}

pub async fn execute(
    local: bool,
    detailed: bool,
    server: &str,
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
        _ => {
            print_models_table(&models, detailed);
        }
    }

    Ok(())
}

fn print_models_table(models: &[ModelRecord], detailed: bool) {
    let mut table = Table::new();

    if detailed {
        table.set_header(vec!["Model", "Status", "Size", "Progress", "Path"]);
    } else {
        table.set_header(vec!["Model", "Status", "Size"]);
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
            let path = model
                .local_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_string());
            table.add_row(vec![
                Cell::new(&model.variant).fg(Color::Cyan),
                status_cell(&model.status),
                Cell::new(size).set_alignment(CellAlignment::Right),
                Cell::new(progress).set_alignment(CellAlignment::Right),
                Cell::new(path),
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
    println!();
    println!(
        "{}: Use {} to download more models",
        style("Tip").yellow().bold(),
        style("izwi pull <model>").cyan()
    );
}

fn status_cell(status: &str) -> Cell {
    let color = match status {
        "ready" | "downloaded" => Color::Green,
        "downloading" => Color::Yellow,
        "loading" => Color::Blue,
        "error" => Color::Red,
        "not_downloaded" => Color::DarkGrey,
        _ => Color::DarkGrey,
    };
    Cell::new(status).fg(color)
}
