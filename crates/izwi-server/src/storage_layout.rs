//! Shared storage layout and filesystem helpers for server persistence.

use anyhow::{Context, anyhow};
use std::path::{Component, Path, PathBuf};

const APP_NAME_DIR: &str = "izwi";
const DEFAULT_DB_FILENAME: &str = "izwi.sqlite3";
const DB_ENV_PRIMARY: &str = "IZWI_DB_PATH";
const MEDIA_ENV_PRIMARY: &str = "IZWI_MEDIA_DIR";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaGroup {
    Uploads,
    Generated,
}

impl MediaGroup {
    pub const fn as_dir(self) -> &'static str {
        match self {
            Self::Uploads => "uploads",
            Self::Generated => "generated",
        }
    }
}

pub fn resolve_db_path() -> PathBuf {
    if let Some(path) = env_path(DB_ENV_PRIMARY) {
        return path;
    }

    resolve_data_root().join(DEFAULT_DB_FILENAME)
}

pub fn resolve_media_root() -> PathBuf {
    if let Some(path) = env_path(MEDIA_ENV_PRIMARY) {
        return path;
    }

    resolve_data_root().join("media")
}

pub fn ensure_storage_dirs(db_path: &Path, media_root: &Path) -> anyhow::Result<()> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create SQLite parent directory: {}",
                parent.display()
            )
        })?;
    }

    ensure_media_dirs(media_root)
}

pub fn ensure_media_dirs(media_root: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(media_root).with_context(|| {
        format!(
            "Failed to create media root directory: {}",
            media_root.display()
        )
    })?;

    for child in ["images", "videos"] {
        let child_dir = media_root.join(child);
        std::fs::create_dir_all(&child_dir).with_context(|| {
            format!(
                "Failed to create media child directory: {}",
                child_dir.display()
            )
        })?;
    }

    Ok(())
}

pub fn persist_audio_file(
    media_root: &Path,
    group: MediaGroup,
    namespace: &str,
    record_id: &str,
    preferred_filename: Option<&str>,
    mime_type: &str,
    bytes: &[u8],
) -> anyhow::Result<String> {
    if bytes.is_empty() {
        return Err(anyhow!("Audio payload cannot be empty"));
    }

    let extension = resolve_audio_extension(preferred_filename, mime_type);
    let relative_path = PathBuf::from(group.as_dir())
        .join(namespace)
        .join(format!("{record_id}.{extension}"));
    let full_path = media_root.join(&relative_path);
    let temp_path = media_root.join(
        PathBuf::from(group.as_dir())
            .join(namespace)
            .join(format!("{record_id}.{}.tmp", uuid::Uuid::new_v4().simple())),
    );

    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create media directory: {}", parent.display()))?;
    }

    std::fs::write(&temp_path, bytes).with_context(|| {
        format!(
            "Failed writing media file to temporary path: {}",
            temp_path.display()
        )
    })?;

    if let Err(err) = std::fs::rename(&temp_path, &full_path) {
        if full_path.exists() {
            std::fs::remove_file(&full_path).with_context(|| {
                format!(
                    "Failed replacing existing media file: {}",
                    full_path.display()
                )
            })?;
            std::fs::rename(&temp_path, &full_path).with_context(|| {
                format!(
                    "Failed moving media file from '{}' to '{}': {err}",
                    temp_path.display(),
                    full_path.display()
                )
            })?;
        } else {
            return Err(err).with_context(|| {
                format!(
                    "Failed moving media file from '{}' to '{}'",
                    temp_path.display(),
                    full_path.display()
                )
            });
        }
    }

    Ok(normalize_relative_path(relative_path))
}

pub fn read_media_file(media_root: &Path, relative_path: &str) -> anyhow::Result<Vec<u8>> {
    let absolute_path = resolve_media_path(media_root, relative_path)?;
    std::fs::read(&absolute_path)
        .with_context(|| format!("Failed to read media file: {}", absolute_path.display()))
}

pub fn content_type_from_media_path(path: &str) -> &'static str {
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        Some("avif") => "image/avif",
        Some("heic") => "image/heic",
        Some("heif") => "image/heif",
        Some("mp4") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        Some("avi") => "video/x-msvideo",
        Some("mkv") => "video/x-matroska",
        Some("mpeg") | Some("mpg") => "video/mpeg",
        Some("3gp") => "video/3gpp",
        Some("wav") => "audio/wav",
        Some("mp3") => "audio/mpeg",
        Some("ogg") => "audio/ogg",
        Some("flac") => "audio/flac",
        Some("m4a") => "audio/mp4",
        Some("aac") => "audio/aac",
        _ => "application/octet-stream",
    }
}

pub fn delete_media_file(media_root: &Path, relative_path: Option<&str>) -> anyhow::Result<()> {
    let Some(relative_path) = relative_path else {
        return Ok(());
    };

    let absolute_path = resolve_media_path(media_root, relative_path)?;
    match std::fs::remove_file(&absolute_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err)
            .with_context(|| format!("Failed deleting media file: {}", absolute_path.display())),
    }
}

fn resolve_data_root() -> PathBuf {
    if let Some(mut dir) = dirs::data_local_dir() {
        dir.push(APP_NAME_DIR);
        return dir;
    }

    PathBuf::from("data")
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key).ok().and_then(|raw| {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    })
}

fn resolve_audio_extension(preferred_filename: Option<&str>, mime_type: &str) -> String {
    if let Some(ext) = preferred_filename
        .and_then(|name| Path::new(name).extension())
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.trim().to_ascii_lowercase())
        .filter(|ext| is_safe_extension(ext.as_str()))
    {
        return ext;
    }

    let mime = mime_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();

    let mapped = match mime.as_str() {
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/ogg" | "audio/vorbis" => "ogg",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/x-m4a" | "audio/m4a" => "m4a",
        "audio/aac" => "aac",
        "audio/basic" => "au",
        _ => "bin",
    };

    mapped.to_string()
}

fn is_safe_extension(ext: &str) -> bool {
    !ext.is_empty()
        && ext.len() <= 12
        && ext
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
}

pub fn resolve_media_path(media_root: &Path, relative_path: &str) -> anyhow::Result<PathBuf> {
    let candidate = Path::new(relative_path);
    if candidate.is_absolute() {
        return Err(anyhow!(
            "Absolute media paths are not allowed: {relative_path}"
        ));
    }

    for component in candidate.components() {
        if matches!(component, Component::ParentDir | Component::Prefix(_)) {
            return Err(anyhow!("Unsafe media path component in '{relative_path}'"));
        }
    }

    Ok(media_root.join(candidate))
}

fn normalize_relative_path(path: PathBuf) -> String {
    path.to_string_lossy().replace('\\', "/")
}
