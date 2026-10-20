//! Desktop-only preferences in `desktop-settings.json`, next to the app's
//! other config. Each preference is one top-level key; writing one keeps the
//! others.

use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::Value;
use tauri::{AppHandle, Manager, Runtime};

const DESKTOP_SETTINGS_FILE: &str = "desktop-settings.json";

/// Serializes read-modify-write cycles from the tray command and the window.
static WRITE_LOCK: Mutex<()> = Mutex::new(());

pub fn read<R: Runtime>(app: &AppHandle<R>, key: &str) -> Option<Value> {
    let contents = std::fs::read_to_string(settings_path(app)?).ok()?;
    value_in(&contents, key)
}

pub fn write<R: Runtime>(app: &AppHandle<R>, key: &str, value: Value) -> std::io::Result<()> {
    let path = settings_path(app).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "app config directory is unavailable",
        )
    })?;
    let _guard = WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let existing = std::fs::read_to_string(&path).ok();
    let contents = merge(existing.as_deref(), key, value);
    // Write then rename so a crash never leaves a truncated settings file.
    let temp_path = path.with_extension("json.tmp");
    std::fs::write(&temp_path, contents)?;
    std::fs::rename(temp_path, path)
}

fn settings_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|dir| dir.join(DESKTOP_SETTINGS_FILE))
}

fn value_in(contents: &str, key: &str) -> Option<Value> {
    serde_json::from_str::<Value>(contents)
        .ok()?
        .get(key)
        .cloned()
}

/// Set `key` while keeping any other keys in the settings file.
fn merge(existing: Option<&str>, key: &str, value: Value) -> String {
    let mut settings = existing
        .and_then(|contents| serde_json::from_str::<Value>(contents).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    settings[key] = value;
    serde_json::to_string_pretty(&settings).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::{merge, value_in};
    use serde_json::{json, Value};

    #[test]
    fn reads_a_saved_key() {
        let contents = r#"{"tray_icon_visible": false}"#;
        assert_eq!(
            value_in(contents, "tray_icon_visible"),
            Some(Value::Bool(false))
        );
        assert_eq!(value_in(contents, "other"), None);
        assert_eq!(value_in("not json", "tray_icon_visible"), None);
    }

    #[test]
    fn merging_keeps_other_settings() {
        let merged = merge(Some(r#"{"other": 1}"#), "tray_icon_visible", json!(false));
        assert_eq!(
            value_in(&merged, "tray_icon_visible"),
            Some(Value::Bool(false))
        );
        assert_eq!(value_in(&merged, "other"), Some(json!(1)));
    }

    #[test]
    fn merging_replaces_unreadable_files() {
        let merged = merge(Some("[1, 2]"), "tray_icon_visible", json!(true));
        assert_eq!(
            value_in(&merged, "tray_icon_visible"),
            Some(Value::Bool(true))
        );
        let merged = merge(None, "main_window", json!({"width": 1}));
        assert_eq!(value_in(&merged, "main_window"), Some(json!({"width": 1})));
    }
}
