use crate::error::{CliError, Result};
use std::path::Path;

/// The file name an audio upload carries: the path's basename.
pub fn audio_filename_from_path(path: &Path) -> Option<String> {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(str::trim)
        .filter(|value| !value.is_empty() && !value.contains(['\r', '\n']))
        .map(ToOwned::to_owned)
}

/// The MIME type of an audio file, from its extension.
pub fn audio_mime_type_from_path(path: &Path) -> Option<&'static str> {
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

/// Reads `path` into the `file` part of a multipart upload, named and typed
/// after the file.
pub async fn audio_file_part(path: &Path) -> Result<reqwest::multipart::Part> {
    let bytes = tokio::fs::read(path).await.map_err(|error| {
        CliError::InvalidInput(format!("Could not read {}: {error}", path.display()))
    })?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(audio_filename_from_path(path).unwrap_or_else(|| "audio".to_string()));
    match audio_mime_type_from_path(path) {
        Some(mime) => part
            .mime_str(mime)
            .map_err(|error| CliError::Other(error.to_string())),
        None => Ok(part),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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
}
