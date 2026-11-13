use anyhow::{Context, Result};
use serde_json::json;
use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, Runtime,
};

pub const TRAY_ID: &str = "izwi-tray";
const TRAY_OPEN_ID: &str = "tray_open";
const TRAY_SETTINGS_ID: &str = "tray_settings";
const TRAY_MODELS_ID: &str = "tray_models";
const TRAY_CHECK_UPDATES_ID: &str = "tray_check_updates";
const TRAY_QUIT_ID: &str = "tray_quit";
const TRAY_DICTATE_ID: &str = "tray_dictate";
const TRAY_PASTE_LAST_ID: &str = "tray_paste_last";

pub fn build_basic_tray<R: Runtime>(app: &AppHandle<R>) -> Result<()> {
    let open_item = MenuItem::with_id(app, TRAY_OPEN_ID, "Open Izwi", true, None::<&str>)
        .context("failed to build tray open item")?;
    let settings_item = MenuItem::with_id(app, TRAY_SETTINGS_ID, "Settings", true, None::<&str>)
        .context("failed to build tray settings item")?;
    let models_item = MenuItem::with_id(app, TRAY_MODELS_ID, "Models", true, None::<&str>)
        .context("failed to build tray models item")?;
    let dictate_item = MenuItem::with_id(
        app,
        TRAY_DICTATE_ID,
        "Start or Stop Dictation",
        true,
        None::<&str>,
    )
    .context("failed to build tray dictation item")?;
    let paste_last_item = MenuItem::with_id(
        app,
        TRAY_PASTE_LAST_ID,
        "Paste Last Dictation",
        true,
        None::<&str>,
    )
    .context("failed to build tray paste item")?;
    let dictation_separator =
        PredefinedMenuItem::separator(app).context("failed to build tray separator item")?;
    let check_updates_item = MenuItem::with_id(
        app,
        TRAY_CHECK_UPDATES_ID,
        "Check for Updates",
        true,
        None::<&str>,
    )
    .context("failed to build tray check updates item")?;
    let separator =
        PredefinedMenuItem::separator(app).context("failed to build tray separator item")?;
    let quit_item = MenuItem::with_id(app, TRAY_QUIT_ID, "Quit Izwi", true, None::<&str>)
        .context("failed to build tray quit item")?;
    let menu = Menu::with_items(
        app,
        &[
            &open_item,
            &dictate_item,
            &paste_last_item,
            &dictation_separator,
            &settings_item,
            &models_item,
            &check_updates_item,
            &separator,
            &quit_item,
        ],
    )
    .context("failed to build tray menu")?;

    let mut tray_builder = TrayIconBuilder::with_id(TRAY_ID)
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(|app_handle, event| match event.id().0.as_str() {
            TRAY_OPEN_ID => {
                let _ = show_main_window(app_handle);
            }
            TRAY_SETTINGS_ID => {
                let _ = navigate_main_window(app_handle, "/settings");
            }
            TRAY_MODELS_ID => {
                let _ = navigate_main_window(app_handle, "/models");
            }
            TRAY_CHECK_UPDATES_ID => {
                let _ = show_main_window(app_handle);
                let _ = emit_tray_event(app_handle, "izwi:tray-check-updates", None);
            }
            TRAY_DICTATE_ID => dictate_from_tray(app_handle),
            TRAY_PASTE_LAST_ID => paste_last_from_tray(app_handle),
            TRAY_QUIT_ID => {
                app_handle.exit(0);
            }
            _ => {}
        });

    if let Some(icon) = app.default_window_icon() {
        tray_builder = tray_builder.icon(icon.clone());
    }

    tray_builder
        .build(app)
        .context("failed to create tray icon")?;

    Ok(())
}

/// Whether the desktop shows tray icons. GNOME shows none unless an
/// AppIndicator extension provides a StatusNotifierWatcher; hiding the window
/// in a tray nobody can see would leave Izwi running with no way back to it.
pub fn desktop_shows_tray_icons() -> bool {
    #[cfg(target_os = "linux")]
    if std::env::var("XDG_CURRENT_DESKTOP").is_ok_and(|desktop| is_gnome(&desktop)) {
        return status_notifier_watcher_running();
    }
    true
}

#[cfg(any(target_os = "linux", test))]
fn is_gnome(xdg_current_desktop: &str) -> bool {
    xdg_current_desktop
        .split(':')
        .any(|name| name.eq_ignore_ascii_case("gnome"))
}

#[cfg(target_os = "linux")]
fn status_notifier_watcher_running() -> bool {
    std::process::Command::new("gdbus")
        .args([
            "call",
            "--session",
            "--timeout",
            "2",
            "--dest",
            "org.freedesktop.DBus",
            "--object-path",
            "/org/freedesktop/DBus",
            "--method",
            "org.freedesktop.DBus.NameHasOwner",
            "org.kde.StatusNotifierWatcher",
        ])
        .output()
        .is_ok_and(|output| {
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("true")
        })
}

/// The dictation tray items act on the real app; test runtimes have none.
fn dictate_from_tray<R: Runtime>(app: &AppHandle<R>) {
    let Some(app) = (app as &dyn std::any::Any).downcast_ref::<AppHandle>() else {
        return;
    };
    if super::dictation::is_enabled(app) {
        super::dictation::send(app, izwi_dictation::Input::Toggle);
    } else {
        // Off: the Dictation page is where it is turned on.
        let _ = navigate_main_window(app, "/dictation");
    }
}

fn paste_last_from_tray<R: Runtime>(app: &AppHandle<R>) {
    if let Some(app) = (app as &dyn std::any::Any).downcast_ref::<AppHandle>() {
        super::dictation::paste_last(app);
    }
}

/// Show in the tray that a dictation is being recorded.
pub fn set_dictation_recording<R: Runtime>(app: &AppHandle<R>, recording: bool) {
    let Some(tray) = app.tray_by_id(TRAY_ID) else {
        return;
    };
    let _ = tray.set_tooltip(Some(if recording { "Izwi: listening" } else { "Izwi" }));
    // The menu bar shows text next to the icon; other trays have no room.
    #[cfg(target_os = "macos")]
    let _ = tray.set_title(if recording { Some("●") } else { None::<&str> });
}

pub fn show_main_window<R: Runtime>(app: &AppHandle<R>) -> Result<()> {
    let window = app
        .get_webview_window("main")
        .context("main window is unavailable")?;
    let _ = window.show();
    let _ = window.unminimize();
    let _ = window.set_focus();
    Ok(())
}

fn navigate_main_window<R: Runtime>(app: &AppHandle<R>, route: &str) -> Result<()> {
    show_main_window(app)?;
    let detail = json!({ "path": route }).to_string();
    emit_tray_event(app, "izwi:tray-route", Some(detail.as_str()))
}

fn emit_tray_event<R: Runtime>(
    app: &AppHandle<R>,
    event_name: &str,
    detail_json: Option<&str>,
) -> Result<()> {
    let window = app
        .get_webview_window("main")
        .context("main window is unavailable")?;

    let js_event_name = js_string_literal(event_name);
    let script = if let Some(detail_json) = detail_json {
        format!(
            "window.dispatchEvent(new CustomEvent({}, {{ detail: {} }}));",
            js_event_name, detail_json
        )
    } else {
        format!("window.dispatchEvent(new Event({}));", js_event_name)
    };

    window
        .eval(script)
        .context("failed to dispatch tray event to frontend")
}

fn js_string_literal(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("\"{}\"", escaped)
}

#[cfg(test)]
mod tests {
    use super::is_gnome;

    #[test]
    fn recognizes_gnome_sessions() {
        assert!(is_gnome("GNOME"));
        assert!(is_gnome("ubuntu:GNOME"));
        assert!(is_gnome("Pop:GNOME"));
        assert!(!is_gnome("KDE"));
        assert!(!is_gnome("XFCE"));
        assert!(!is_gnome(""));
    }
}
