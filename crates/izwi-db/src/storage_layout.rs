//! Shared storage layout and filesystem helpers for server persistence.

use anyhow::{anyhow, Context};
use std::path::{Component, Path, PathBuf};

const APP_NAME_DIR: &str = "izwi";
const DEFAULT_DB_FILENAME: &str = "izwi.sqlite3";
const DB_ENV_PRIMARY: &str = "IZWI_DB_PATH";
const DATABASE_URL_ENV: &str = "IZWI_DATABASE_URL";
const MEDIA_ENV_PRIMARY: &str = "IZWI_MEDIA_DIR";
const SPEECH_SPOOL_ENV_PRIMARY: &str = "IZWI_SPEECH_SPOOL_DIR";
const MAX_DATABASE_URL_BYTES: usize = 4096;

/// Database backends the store layer can address by URL. SQLite paths stay
/// the default resolution; the URL form exists for fleet deployments where
/// the durable store lives in a shared server database (DS5).
pub const DATABASE_URL_SCHEMES: &[&str] = &["postgres", "postgresql", "sqlite"];

/// Where the store database lives: a SQLite file path (the historical and
/// default form) or a bounded database URL such as `postgres://...`.
#[derive(Clone, PartialEq, Eq)]
pub enum DatabaseSource {
    Path(PathBuf),
    Url(String),
}

impl std::fmt::Debug for DatabaseSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Path(path) => f.debug_tuple("Path").field(path).finish(),
            Self::Url(url) => f
                .debug_tuple("Url")
                .field(&redact_database_url(url))
                .finish(),
        }
    }
}

/// `text` (a driver error, say) with `url` and its credentials masked.
pub fn redact_database_url_in(text: &str, url: &str) -> String {
    let mut redacted = text.replace(url, &redact_database_url(url));
    for secret in database_url_secrets(url) {
        // Short values would mask unrelated text; the full URL is masked
        // above in any case.
        if secret.len() >= 4 {
            redacted = redacted.replace(secret, "***");
        }
    }
    redacted
}

fn database_url_secrets(url: &str) -> Vec<&str> {
    let Some((_, rest)) = url.split_once("://") else {
        return Vec::new();
    };
    let (location, query) = rest.split_once('?').unwrap_or((rest, ""));
    let mut secrets = Vec::new();
    if let Some((userinfo, _)) = location.rsplit_once('@') {
        if let Some((_, password)) = userinfo.split_once(':') {
            secrets.push(password);
        }
    }
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            if is_secret_query_key(key) {
                secrets.push(value);
            }
        }
    }
    secrets
}

fn is_secret_query_key(key: &str) -> bool {
    ["password", "pass", "pwd", "sslpassword", "passfile"]
        .contains(&key.to_ascii_lowercase().as_str())
}

/// `url` with its password and any credential query values masked, for
/// errors, logs and `Debug` output.
pub fn redact_database_url(url: &str) -> String {
    const MASK: &str = "***";
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_string();
    };
    let (location, query) = match rest.split_once('?') {
        Some((location, query)) => (location, Some(query)),
        None => (rest, None),
    };
    // The last `@` ends the user information; a password may contain more.
    let location = match location.rsplit_once('@') {
        Some((userinfo, host)) => match userinfo.split_once(':') {
            Some((user, _)) => format!("{user}:{MASK}@{host}"),
            None => format!("{userinfo}@{host}"),
        },
        None => location.to_string(),
    };
    let query = query.map(|query| {
        query
            .split('&')
            .map(|pair| match pair.split_once('=') {
                Some((key, _)) if is_secret_query_key(key) => {
                    format!("{key}={MASK}")
                }
                _ => pair.to_string(),
            })
            .collect::<Vec<_>>()
            .join("&")
    });
    match query {
        Some(query) => format!("{scheme}://{location}?{query}"),
        None => format!("{scheme}://{location}"),
    }
}

/// Parse a raw database reference. Values carrying a recognized `scheme://`
/// prefix become [`DatabaseSource::Url`]; anything else is treated as a
/// file path. Unknown URL schemes are rejected instead of being silently
/// downgraded to paths so a mistyped URL fails loudly.
pub fn database_source_from_raw(raw: &str) -> anyhow::Result<DatabaseSource> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("Database reference must not be empty"));
    }
    if trimmed.len() > MAX_DATABASE_URL_BYTES {
        return Err(anyhow!(
            "Database reference exceeds its {MAX_DATABASE_URL_BYTES} byte limit"
        ));
    }
    if let Some((scheme, rest)) = trimmed.split_once("://") {
        if !DATABASE_URL_SCHEMES.contains(&scheme) || rest.is_empty() {
            return Err(anyhow!(
                "Database URL scheme must be one of: {}",
                DATABASE_URL_SCHEMES.join(", ")
            ));
        }
        return Ok(DatabaseSource::Url(trimmed.to_string()));
    }
    Ok(DatabaseSource::Path(PathBuf::from(trimmed)))
}

/// Resolve the durable store database source. `IZWI_DATABASE_URL` wins over
/// the historical `IZWI_DB_PATH` file path; the default remains the local
/// SQLite layout under the data root. An empty URL value means unset.
pub fn resolve_database_source() -> anyhow::Result<DatabaseSource> {
    if let Ok(raw) = std::env::var(DATABASE_URL_ENV) {
        if !raw.trim().is_empty() {
            return database_source_from_raw(&raw);
        }
    }
    Ok(DatabaseSource::Path(resolve_db_path()))
}

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

/// Resolve the process-owned scratch root used by local speech requests.
///
/// This is intentionally separate from durable media: files below this root
/// are temporary and can be reclaimed only through the ownership protocol in
/// `speech_spool`.
pub fn resolve_speech_spool_root() -> PathBuf {
    if let Some(path) = env_path(SPEECH_SPOOL_ENV_PRIMARY) {
        return path;
    }

    std::env::temp_dir().join(APP_NAME_DIR).join("speech-spool")
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

    let parent = full_path
        .parent()
        .context("Media file path has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("Failed to create media directory: {}", parent.display()))?;

    let persisted = write_file_durably(&temp_path, &full_path, bytes);
    if persisted.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    persisted?;

    Ok(normalize_relative_path(relative_path))
}

/// Write `bytes` to `temp_path`, flush them to disk, move the file over
/// `full_path` and flush the directory entry, so a crash leaves either the
/// old file or the complete new one.
fn write_file_durably(temp_path: &Path, full_path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    let mut file = std::fs::File::create(temp_path).with_context(|| {
        format!(
            "Failed writing media file to temporary path: {}",
            temp_path.display()
        )
    })?;
    file.write_all(bytes)
        .and_then(|()| file.sync_all())
        .with_context(|| {
            format!(
                "Failed writing media file to temporary path: {}",
                temp_path.display()
            )
        })?;
    drop(file);

    // rename replaces an existing file atomically on Unix. Windows refuses
    // to rename over an existing file, so there the old file goes first.
    if let Err(err) = std::fs::rename(temp_path, full_path) {
        if !full_path.is_file() {
            return Err(err).with_context(|| {
                format!(
                    "Failed moving media file from '{}' to '{}'",
                    temp_path.display(),
                    full_path.display()
                )
            });
        }
        std::fs::remove_file(full_path).with_context(|| {
            format!(
                "Failed replacing existing media file: {}",
                full_path.display()
            )
        })?;
        std::fs::rename(temp_path, full_path).with_context(|| {
            format!(
                "Failed moving media file from '{}' to '{}': {err}",
                temp_path.display(),
                full_path.display()
            )
        })?;
    }
    #[cfg(unix)]
    if let Some(parent) = full_path.parent() {
        std::fs::File::open(parent)
            .and_then(|directory| directory.sync_all())
            .with_context(|| format!("Failed to sync media directory: {}", parent.display()))?;
    }
    Ok(())
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
        Some("f32le") => "audio/pcm-f32le",
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

    media_extension_for_content_type(&mime).to_string()
}

/// Stable extension mapping used by local providers that reconstruct MIME
/// metadata from object keys on read.
pub fn media_extension_for_content_type(mime_type: &str) -> &'static str {
    let mime = mime_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match mime.as_str() {
        "audio/wav" | "audio/x-wav" | "audio/wave" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/ogg" | "audio/vorbis" => "ogg",
        "audio/flac" | "audio/x-flac" => "flac",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/x-m4a" | "audio/m4a" => "m4a",
        "audio/aac" => "aac",
        "audio/pcm-f32le" => "f32le",
        "audio/basic" => "au",
        "image/jpeg" => "jpg",
        "image/png" => "png",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/bmp" => "bmp",
        "image/svg+xml" => "svg",
        "image/avif" => "avif",
        "image/heic" => "heic",
        "image/heif" => "heif",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        "video/x-msvideo" => "avi",
        "video/x-matroska" => "mkv",
        "video/mpeg" => "mpeg",
        "video/3gpp" => "3gp",
        _ => "bin",
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mysql_database_urls_are_rejected() {
        let error = database_source_from_raw("mysql://izwi@db.internal/izwi")
            .expect_err("MySQL is not a supported backend");
        assert!(error.to_string().contains("postgres, postgresql, sqlite"), "{error}");
        assert!(matches!(
            database_source_from_raw("postgres://izwi@db.internal/izwi"),
            Ok(DatabaseSource::Url(_))
        ));
    }

    #[test]
    fn database_urls_are_redacted_in_debug_output() {
        assert_eq!(
            redact_database_url(
                "postgres://izwi:s3cr@t@db.internal:5432/izwi?sslmode=require&password=other"
            ),
            "postgres://izwi:***@db.internal:5432/izwi?sslmode=require&password=***"
        );
        assert_eq!(
            redact_database_url("postgres://izwi@db.internal/izwi"),
            "postgres://izwi@db.internal/izwi"
        );
        assert_eq!(
            redact_database_url("sqlite:///var/lib/izwi/izwi.sqlite3?mode=rwc"),
            "sqlite:///var/lib/izwi/izwi.sqlite3?mode=rwc"
        );
        let source = DatabaseSource::Url("postgresql://u:hunter2@h/db?PassWord=x".to_string());
        let debug = format!("{source:?}");
        assert!(
            !debug.contains("hunter2") && !debug.contains("=x"),
            "{debug}"
        );
    }

    #[test]
    fn persisted_media_replaces_the_target_and_never_leaves_temporary_files() {
        let root = tempfile::tempdir().unwrap();
        let key = persist_audio_file(
            root.path(),
            MediaGroup::Uploads,
            "uploads",
            "record",
            Some("clip.wav"),
            "audio/wav",
            b"first",
        )
        .unwrap();
        persist_audio_file(
            root.path(),
            MediaGroup::Uploads,
            "uploads",
            "record",
            Some("clip.wav"),
            "audio/wav",
            b"second",
        )
        .unwrap();
        assert_eq!(read_media_file(root.path(), &key).unwrap(), b"second");

        // A target that cannot be replaced fails without leaking the
        // temporary file.
        let blocked = resolve_media_path(root.path(), &key).unwrap();
        std::fs::remove_file(&blocked).unwrap();
        std::fs::create_dir(&blocked).unwrap();
        std::fs::write(blocked.join("occupied"), b"x").unwrap();
        assert!(persist_audio_file(
            root.path(),
            MediaGroup::Uploads,
            "uploads",
            "record",
            Some("clip.wav"),
            "audio/wav",
            b"third",
        )
        .is_err());
        let leftovers = std::fs::read_dir(blocked.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn raw_f32_pcm_content_type_round_trips_through_local_object_keys() {
        let extension = media_extension_for_content_type("audio/pcm-f32le");
        assert_eq!(extension, "f32le");
        assert_eq!(
            content_type_from_media_path(&format!("chunk.{extension}")),
            "audio/pcm-f32le"
        );
    }
}
