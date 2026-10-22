//! Saving files the user exports. The webview cancels `<a download>` clicks
//! (no download handler is set), so the desktop app asks where to save with a
//! native dialog and writes the file itself.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use tauri::ipc::{InvokeBody, Request};
use tauri::WebviewWindow;
use tauri_plugin_dialog::DialogExt;
use url::Url;

use super::server::is_local_server_host;

/// Header carrying the suggested file name of a `save_file` call,
/// URI-encoded because header values are ASCII.
const FILENAME_HEADER: &str = "x-izwi-filename";
const FALLBACK_FILENAME: &str = "export.txt";

/// Save the bytes in the request body where the user chooses. Returns the
/// saved path, or None when the save dialog was cancelled.
#[tauri::command]
pub async fn save_file(
    window: WebviewWindow,
    request: Request<'_>,
) -> Result<Option<String>, String> {
    let bytes = match request.body() {
        InvokeBody::Raw(bytes) => bytes.clone(),
        // The postMessage fallback sends a byte array as JSON.
        InvokeBody::Json(value) => serde_json::from_value::<Vec<u8>>(value.clone())
            .map_err(|_| "save_file expects the file's bytes".to_string())?,
    };
    let suggested = request
        .headers()
        .get(FILENAME_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(|value| {
            percent_encoding::percent_decode_str(value)
                .decode_utf8_lossy()
                .into_owned()
        })
        .unwrap_or_default();

    let Some(path) = choose_save_path(&window, &suggested, FALLBACK_FILENAME).await? else {
        return Ok(None);
    };
    let target = path.clone();
    tauri::async_runtime::spawn_blocking(move || {
        std::fs::write(&target, bytes)
            .with_context(|| format!("failed writing {}", target.display()))
    })
    .await
    .map_err(|error| format!("save task failed: {error}"))?
    .map_err(|error| format!("{error:#}"))?;
    Ok(Some(path.to_string_lossy().into_owned()))
}

/// Save a file served by the local Izwi server (audio) where the user
/// chooses. Returns the saved path, or None when the dialog was cancelled.
#[tauri::command]
pub async fn download_audio_file(
    window: WebviewWindow,
    url: String,
    suggested_filename: Option<String>,
) -> Result<Option<String>, String> {
    let url = local_server_url(&url).map_err(|error| error.to_string())?;
    let suggested = suggested_filename.unwrap_or_default();
    let Some(path) = choose_save_path(&window, &suggested, "speech.wav").await? else {
        return Ok(None);
    };
    download_to(url, &path)
        .await
        .map_err(|error| format!("{error:#}"))?;
    Ok(Some(path.to_string_lossy().into_owned()))
}

fn local_server_url(url: &str) -> Result<Url> {
    let parsed = Url::parse(url).with_context(|| format!("invalid audio URL: {url}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        anyhow::bail!("unsupported audio URL scheme: {}", parsed.scheme());
    }
    let host = parsed.host_str().context("audio URL is missing host")?;
    if !is_local_server_host(host) {
        anyhow::bail!("audio download is allowed only from local Izwi server URLs");
    }
    Ok(parsed)
}

/// Stream `url` into `path` through a partial file, so a failed download
/// neither leaves a truncated file nor replaces the one the user picked.
async fn download_to(url: Url, path: &Path) -> Result<()> {
    // reqwest's rustls has no built-in crypto provider in this workspace.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut response = reqwest::get(url.clone())
        .await
        .with_context(|| format!("failed downloading {url}"))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        // Prefer the server's reason ("Render these paragraphs…") to a status.
        return Err(anyhow!(server_error_message(&body)
            .unwrap_or_else(|| format!("Audio download failed ({status})"))));
    }

    let partial = partial_path(path);
    let result = async {
        let mut file = tokio::fs::File::create(&partial)
            .await
            .with_context(|| format!("failed creating {}", partial.display()))?;
        while let Some(chunk) = response
            .chunk()
            .await
            .context("failed reading downloaded audio")?
        {
            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .with_context(|| format!("failed writing {}", partial.display()))?;
        }
        tokio::io::AsyncWriteExt::flush(&mut file).await?;
        drop(file);
        tokio::fs::rename(&partial, path)
            .await
            .with_context(|| format!("failed saving {}", path.display()))
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&partial).await;
    }
    result
}

fn partial_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    path.with_file_name(name)
}

fn server_error_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    value
        .pointer("/error/message")
        .and_then(|message| message.as_str())
        .map(str::to_owned)
        .filter(|message| !message.trim().is_empty())
}

/// Ask where to save, suggesting `suggested` (or `fallback`) and filtering
/// on its extension. None when the user cancels.
async fn choose_save_path(
    window: &WebviewWindow,
    suggested: &str,
    fallback: &str,
) -> Result<Option<PathBuf>, String> {
    let filename = sanitize_download_filename(suggested, fallback);
    let extension = Path::new(&filename)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_owned);

    let mut dialog = window
        .dialog()
        .file()
        .set_file_name(&filename)
        .set_parent(window);
    if let Some(extension) = extension.as_deref() {
        dialog = dialog.add_filter(file_type_label(extension), &[extension]);
    }
    let (sender, receiver) = tokio::sync::oneshot::channel();
    dialog.save_file(move |path| {
        let _ = sender.send(path);
    });
    let Some(path) = receiver
        .await
        .map_err(|_| "the save dialog closed unexpectedly".to_string())?
    else {
        return Ok(None);
    };
    let path = path.into_path().map_err(|error| error.to_string())?;
    // Saved under the name the user confirmed: adding an extension here
    // could replace a file the dialog never asked about.
    Ok(Some(path))
}

fn file_type_label(extension: &str) -> String {
    match extension.to_ascii_lowercase().as_str() {
        "txt" => "Text".to_string(),
        "json" => "JSON".to_string(),
        "srt" => "SubRip subtitles".to_string(),
        "vtt" => "WebVTT subtitles".to_string(),
        "wav" => "WAV audio".to_string(),
        "mp3" => "MP3 audio".to_string(),
        "flac" => "FLAC audio".to_string(),
        "ogg" | "opus" => "Ogg audio".to_string(),
        other => format!("{} file", other.to_ascii_uppercase()),
    }
}

/// Replace characters no file system accepts; keep the rest, accents
/// included.
fn sanitize_download_filename(raw: &str, fallback: &str) -> String {
    let sanitized: String = raw
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_control() || matches!(ch, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
            {
                '_'
            } else {
                ch
            }
        })
        .collect();
    let sanitized = sanitized.trim().trim_matches('.').trim();
    if sanitized.is_empty() {
        fallback.to_string()
    } else {
        sanitized.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_download_filename_replaces_unsafe_chars() {
        assert_eq!(
            sanitize_download_filename("voice:*?/sample.wav", "speech.wav"),
            "voice____sample.wav"
        );
        assert_eq!(
            sanitize_download_filename("Réunion d'équipe.txt", "export.txt"),
            "Réunion d'équipe.txt"
        );
        assert_eq!(
            sanitize_download_filename("  ..  ", "speech.wav"),
            "speech.wav"
        );
    }

    #[test]
    fn audio_downloads_only_come_from_the_local_server() {
        assert!(local_server_url("http://localhost:8080/v1/x/audio").is_ok());
        assert!(local_server_url("http://127.0.0.1:8081/v1/x/audio").is_ok());
        assert!(local_server_url("https://example.com/audio.wav").is_err());
        assert!(local_server_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn server_errors_keep_their_reason() {
        assert_eq!(
            server_error_message(r#"{"error":{"message":"Render these paragraphs first"}}"#)
                .as_deref(),
            Some("Render these paragraphs first")
        );
        assert_eq!(server_error_message("Bad gateway"), None);
    }

    #[test]
    fn partial_downloads_sit_next_to_the_target() {
        assert_eq!(
            partial_path(Path::new("/tmp/take.wav")),
            PathBuf::from("/tmp/take.wav.part")
        );
    }
}
