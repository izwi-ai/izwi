use std::sync::Mutex;

use tauri::{AppHandle, Manager, Runtime, State};

use super::settings;

const TRAY_ICON_VISIBLE_KEY: &str = "tray_icon_visible";

pub struct TrayVisibilityState {
    visible: Mutex<bool>,
    /// False when the desktop shows no tray icons or the tray failed to build.
    available: Mutex<bool>,
}

impl TrayVisibilityState {
    pub fn new(initial_visible: bool) -> Self {
        Self {
            visible: Mutex::new(initial_visible),
            available: Mutex::new(true),
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

    pub fn is_available(&self) -> bool {
        self.available.lock().map(|guard| *guard).unwrap_or(true)
    }

    pub fn set_available(&self, available: bool) {
        if let Ok(mut guard) = self.available.lock() {
            *guard = available;
        }
    }

    /// Closing the window hides it in the tray only when there is a tray
    /// icon to bring it back from; otherwise closing it quits.
    pub fn hides_window_on_close(&self) -> bool {
        self.is_available() && self.is_visible()
    }
}

#[tauri::command]
pub fn tray_icon_visible(state: State<'_, TrayVisibilityState>) -> bool {
    state.hides_window_on_close()
}

#[tauri::command]
pub fn set_tray_icon_visible(
    app: AppHandle,
    state: State<'_, TrayVisibilityState>,
    visible: bool,
) -> std::result::Result<(), String> {
    if !state.is_available() {
        return Err("Your desktop doesn't show tray icons.".to_string());
    }
    let tray = app
        .tray_by_id(super::tray::TRAY_ID)
        .ok_or_else(|| "Tray icon is unavailable".to_string())?;

    tray.set_visible(visible)
        .map_err(|error| error.to_string())?;
    state.set_visible(visible);
    settings::write(
        &app,
        TRAY_ICON_VISIBLE_KEY,
        serde_json::Value::Bool(visible),
    )
    .map_err(|error| format!("Tray setting applied but not saved: {error}"))
}

/// Restore the saved tray preference after the tray is built. The state
/// starts visible because it is created before the app (and its config
/// directory) exists.
pub fn restore_tray_visibility<R: Runtime>(app: &AppHandle<R>) {
    let Some(visible) =
        settings::read(app, TRAY_ICON_VISIBLE_KEY).and_then(|value| value.as_bool())
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

#[cfg(test)]
mod tests {
    use super::TrayVisibilityState;

    #[test]
    fn closing_hides_the_window_only_with_a_visible_available_tray() {
        let state = TrayVisibilityState::new(true);
        assert!(state.hides_window_on_close());

        state.set_visible(false);
        assert!(!state.hides_window_on_close());

        state.set_visible(true);
        state.set_available(false);
        assert!(!state.hides_window_on_close());
    }
}
