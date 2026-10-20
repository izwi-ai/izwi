use anyhow::Result;
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

use super::tray_visibility::TrayVisibilityState;
use super::window_state;

pub struct WindowConfig {
    pub server_origin: String,
    pub window_title: String,
    pub width: f64,
    pub height: f64,
}

pub fn build_main_window<R: tauri::Runtime>(
    app: &mut tauri::App<R>,
    config: &WindowConfig,
) -> Result<()> {
    let init_script = format!(
        "window.__IZWI_SERVER_URL__ = {};",
        js_string_literal(&config.server_origin)
    );
    let saved_geometry = window_state::load(app.handle());
    let mut window_builder =
        WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
            .initialization_script(init_script)
            .title(config.window_title.as_str())
            .inner_size(config.width, config.height)
            .min_inner_size(960.0, 680.0)
            .resizable(true)
            // Tauri's file-drop handler swallows HTML5 drag and drop: files
            // dropped on New transcript and Studio's paragraph reordering.
            .disable_drag_drop_handler()
            // Shown once the saved size and position are applied.
            .visible(saved_geometry.is_none());

    if let Some(icon) = app.default_window_icon() {
        window_builder = window_builder.icon(icon.clone())?;
    }

    let window = window_builder.build()?;
    if let Some(geometry) = saved_geometry {
        window_state::restore(&window, geometry);
        let _ = window.show();
        let _ = window.set_focus();
    }
    Ok(())
}

pub fn handle_run_event<R: tauri::Runtime>(app_handle: &tauri::AppHandle<R>, event: &RunEvent) {
    // Clicking the Dock icon while the window is hidden in the tray should
    // bring it back, as it does for native macOS apps.
    #[cfg(target_os = "macos")]
    if let RunEvent::Reopen {
        has_visible_windows: false,
        ..
    } = event
    {
        let _ = super::tray::show_main_window(app_handle);
    }

    if let RunEvent::ExitRequested { .. } = event {
        if let Some(window) = app_handle.get_webview_window("main") {
            if window.is_visible().unwrap_or(false) {
                window_state::save(&window);
                // Gone at once, while the server stops behind it.
                let _ = window.hide();
            }
        }
    }

    if let RunEvent::WindowEvent { label, event, .. } = event {
        if label == "main" {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let window = app_handle.get_webview_window("main");
                if let Some(window) = &window {
                    window_state::save(window);
                }
                let should_hide = app_handle
                    .state::<TrayVisibilityState>()
                    .hides_window_on_close();
                if should_hide {
                    api.prevent_close();
                    if let Some(window) = window {
                        let _ = window.hide();
                    }
                }
            }
        }
    }
}

pub fn event_ends_desktop_session<R: tauri::Runtime>(
    app_handle: &tauri::AppHandle<R>,
    event: &RunEvent,
) -> bool {
    match event {
        RunEvent::ExitRequested { .. } | RunEvent::Exit => true,
        RunEvent::WindowEvent {
            label,
            event: WindowEvent::Destroyed,
            ..
        } => {
            label == "main"
                && !app_handle
                    .state::<TrayVisibilityState>()
                    .hides_window_on_close()
        }
        _ => false,
    }
}

fn js_string_literal(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r");
    format!("\"{}\"", escaped)
}
