use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager, Runtime, State};

/// Desktop-only preferences, stored next to the app's other config.
const DESKTOP_SETTINGS_FILE: &str = "desktop-settings.json";
const TRAY_ICON_VISIBLE_KEY: &str = "tray_icon_visible";

pub struct TrayVisibilityState {
    visible: Mutex<bool>,
}

impl TrayVisibilityState {
    pub fn new(initial_visible: bool) -> Self {
        Self {
            visible: Mutex::new(initial_visible),
        }
    }

    pub fn is_visible(&self) -> bool {
        self.visible.lock().map(|guard| *guard).unwrap_or(true)
    }

    pub fn set_visible(&self, visible: bool) {
        if let Ok(mut guard) = self.visible.lock() {
            *guard = visible;
        }
    }
}

#[tauri::command]
pub fn tray_icon_visible(state: State<'_, TrayVisibilityState>) -> bool {
    state.is_visible()
}

#[tauri::command]
pub fn set_tray_icon_visible(
    app: AppHandle,
    state: State<'_, TrayVisibilityState>,
    visible: bool,
) -> std::result::Result<(), String> {
    let tray = app
        .tray_by_id(super::tray::TRAY_ID)
        .ok_or_else(|| "Tray icon is unavailable".to_string())?;

    tray.set_visible(visible)
        .map_err(|error| error.to_string())?;
    state.set_visible(visible);
    save_tray_visibility(&app, visible)
        .map_err(|error| format!("Tray setting applied but not saved: {error}"))
}

/// Restore the saved tray preference after the tray is built. The state
/// starts visible because it is created before the app (and its config
/// directory) exists.
pub fn restore_tray_visibility<R: Runtime>(app: &AppHandle<R>) {
    let Some(visible) = settings_path(app)
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|contents| parse_tray_visibility(&contents))
    else {
        return;
    };
    app.state::<TrayVisibilityState>().set_visible(visible);
    if !visible {
        if let Some(tray) = app.tray_by_id(super::tray::TRAY_ID) {
            let _ = tray.set_visible(false);
        }
    }
}

fn settings_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|dir| dir.join(DESKTOP_SETTINGS_FILE))
}

fn save_tray_visibility<R: Runtime>(app: &AppHandle<R>, visible: bool) -> std::io::Result<()> {
    let path = settings_path(app).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "app config directory is unavailable",
        )
    })?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let existing = std::fs::read_to_string(&path).ok();
    let contents = merge_tray_visibility(existing.as_deref(), visible);
    // Write then rename so a crash never leaves a truncated settings file.
    let temp_path = path.with_extension("json.tmp");
    std::fs::write(&temp_path, contents)?;
    std::fs::rename(temp_path, path)
}

fn parse_tray_visibility(contents: &str) -> Option<bool> {
    serde_json::from_str::<serde_json::Value>(contents)
        .ok()?
        .get(TRAY_ICON_VISIBLE_KEY)?
        .as_bool()
}

/// Set the tray key while keeping any other keys in the settings file.
fn merge_tray_visibility(existing: Option<&str>, visible: bool) -> String {
    let mut settings = existing
        .and_then(|contents| serde_json::from_str::<serde_json::Value>(contents).ok())
        .filter(serde_json::Value::is_object)
        .unwrap_or_else(|| serde_json::json!({}));
    settings[TRAY_ICON_VISIBLE_KEY] = serde_json::Value::Bool(visible);
    serde_json::to_string_pretty(&settings).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::{merge_tray_visibility, parse_tray_visibility};

    #[test]
    fn reads_the_saved_tray_preference() {
        assert_eq!(
            parse_tray_visibility(r#"{"tray_icon_visible": false}"#),
            Some(false)
        );
        assert_eq!(
            parse_tray_visibility(r#"{"tray_icon_visible": true}"#),
            Some(true)
        );
        assert_eq!(parse_tray_visibility(r#"{"other": 1}"#), None);
        assert_eq!(parse_tray_visibility("not json"), None);
    }

    #[test]
    fn merging_keeps_other_settings() {
        let merged = merge_tray_visibility(Some(r#"{"other": 1}"#), false);
        assert_eq!(parse_tray_visibility(&merged), Some(false));
        assert!(merged.contains("\"other\": 1"));
    }

    #[test]
    fn merging_replaces_unreadable_files() {
        let merged = merge_tray_visibility(Some("[1, 2]"), true);
        assert_eq!(parse_tray_visibility(&merged), Some(true));
        let merged = merge_tray_visibility(None, false);
        assert_eq!(parse_tray_visibility(&merged), Some(false));
    }
}
