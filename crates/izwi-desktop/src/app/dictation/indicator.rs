//! The dictation indicator: a small pill near the bottom of the screen that
//! shows listening, transcribing and notices. It must never take keyboard
//! focus, or the paste would land in it instead of the user's app.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tauri::{AppHandle, Manager, PhysicalPosition, Runtime, WebviewUrl, WebviewWindowBuilder};

use super::super::desktop_log;

pub const INDICATOR_LABEL: &str = "dictation-indicator";
const WIDTH: f64 = 300.0;
const HEIGHT: f64 = 64.0;
/// Distance from the bottom of the screen's work area.
const BOTTOM_MARGIN: f64 = 96.0;

/// Bumped on every show and hide request; a delayed hide only runs if no
/// newer request arrived.
static GENERATION: AtomicU64 = AtomicU64::new(0);

/// Build the window, hidden. Done once at startup: creating a webview takes
/// too long to do when the hotkey goes down.
pub fn ensure<R: Runtime>(app: &AppHandle<R>) {
    if app.get_webview_window(INDICATOR_LABEL).is_some() {
        return;
    }
    let built = WebviewWindowBuilder::new(
        app,
        INDICATOR_LABEL,
        WebviewUrl::App("indicator.html".into()),
    )
    .title("Izwi Dictation")
    .inner_size(WIDTH, HEIGHT)
    .resizable(false)
    .decorations(false)
    .transparent(true)
    .shadow(false)
    .always_on_top(true)
    .visible_on_all_workspaces(true)
    .skip_taskbar(true)
    .focused(false)
    .focusable(false)
    .visible(false)
    .build();
    match built {
        Ok(window) => {
            let _ = window.set_ignore_cursor_events(true);
        }
        Err(error) => desktop_log::warn(format!("no dictation indicator: {error}")),
    }
}

pub fn destroy<R: Runtime>(app: &AppHandle<R>) {
    GENERATION.fetch_add(1, Ordering::AcqRel);
    if let Some(window) = app.get_webview_window(INDICATOR_LABEL) {
        let _ = window.destroy();
    }
}

/// Show the indicator at the bottom centre of the screen the main window is
/// on (the primary screen when that is unknown).
pub fn show<R: Runtime>(app: &AppHandle<R>) {
    GENERATION.fetch_add(1, Ordering::AcqRel);
    let Some(window) = app.get_webview_window(INDICATOR_LABEL) else {
        return;
    };
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let scale = monitor.scale_factor();
        let area = monitor.work_area();
        let x = f64::from(area.position.x) + (f64::from(area.size.width) - WIDTH * scale) / 2.0;
        let y = f64::from(area.position.y) + f64::from(area.size.height)
            - (HEIGHT + BOTTOM_MARGIN) * scale;
        let _ = window.set_position(PhysicalPosition::new(x.round() as i32, y.round() as i32));
    }
    let _ = window.show();
}

/// Hide the indicator after `delay`, unless it is shown again first.
pub fn hide_after<R: Runtime>(app: &AppHandle<R>, delay: Duration) {
    let generation = GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    let app = app.clone();
    std::thread::spawn(move || {
        if !delay.is_zero() {
            std::thread::sleep(delay);
        }
        if GENERATION.load(Ordering::Acquire) != generation {
            return;
        }
        if let Some(window) = app.get_webview_window(INDICATOR_LABEL) {
            let _ = window.hide();
        }
    });
}
