use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use url::Url;

use super::server::is_local_server_host;

#[tauri::command]
pub async fn download_audio_file(
    url: String,
    suggested_filename: Option<String>,
) -> Result<String, String> {
    let task = tauri::async_runtime::spawn_blocking(move || {
        save_audio_from_url(url.as_str(), suggested_filename.as_deref())
    });

    let saved_path = task
        .await
        .map_err(|err| format!("audio download task failed: {err}"))?
        .map_err(|err| err.to_string())?;

    Ok(saved_path.to_string_lossy().to_string())
}

fn save_audio_from_url(url: &str, suggested_filename: Option<&str>) -> Result<PathBuf> {
    let parsed_url = Url::parse(url).with_context(|| format!("invalid audio URL: {url}"))?;
    if !matches!(parsed_url.scheme(), "http" | "https") {
        anyhow::bail!("unsupported audio URL scheme: {}", parsed_url.scheme());
    }

    let host = parsed_url
        .host_str()
        .context("audio URL is missing host")?
        .to_string();
    if !is_local_server_host(host.as_str()) {
        anyhow::bail!("audio download is allowed only from local Izwi server URLs");
    }

    let response = reqwest::blocking::get(parsed_url.clone())
        .with_context(|| format!("failed downloading audio from {parsed_url}"))?
        .error_for_status()
        .with_context(|| format!("audio download failed for {parsed_url}"))?;
    let bytes = response
        .bytes()
        .context("failed reading downloaded audio bytes")?;

    let filename = sanitize_download_filename(suggested_filename.unwrap_or("speech.wav"));
    let downloads_dir = dirs::download_dir()
        .or_else(|| std::env::current_dir().ok())
        .context("could not determine a downloads directory")?;
    std::fs::create_dir_all(&downloads_dir)
        .with_context(|| format!("failed creating {}", downloads_dir.display()))?;

    let destination = unique_download_path(&downloads_dir, filename.as_str());
    std::fs::write(&destination, bytes.as_ref())
        .with_context(|| format!("failed writing {}", destination.display()))?;

    Ok(destination)
}

fn sanitize_download_filename(raw: &str) -> String {
    let trimmed = raw.trim();
    let source = if trimmed.is_empty() {
        "speech.wav"
    } else {
        trimmed
    };

    let mut sanitized = String::with_capacity(source.len());
    for ch in source.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_' | ' ') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }

    let sanitized = sanitized.trim().trim_matches('.').to_string();
    if sanitized.is_empty() {
        "speech.wav".to_string()
    } else {
        sanitized
    }
}

fn unique_download_path(downloads_dir: &Path, filename: &str) -> PathBuf {
    let first_path = downloads_dir.join(filename);
    if !first_path.exists() {
        return first_path;
    }

    let path = Path::new(filename);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or("speech");
    let ext = path.extension().and_then(|value| value.to_str());

    for index in 1..=10_000usize {
        let candidate_name = match ext {
            Some(ext) if !ext.is_empty() => format!("{stem}-{index}.{ext}"),
            _ => format!("{stem}-{index}"),
        };
        let candidate_path = downloads_dir.join(candidate_name);
        if !candidate_path.exists() {
            return candidate_path;
        }
    }

    downloads_dir.join(format!("{stem}-{}.wav", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::sanitize_download_filename;

    #[test]
    fn sanitize_download_filename_replaces_unsafe_chars() {
        assert_eq!(
            sanitize_download_filename("voice:*?/sample.wav"),
            "voice____sample.wav"
        );
    }
}
