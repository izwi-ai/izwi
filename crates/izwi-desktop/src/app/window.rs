use anyhow::Result;
use tauri::{RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

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
    let mut window_builder =
        WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
            .initialization_script(init_script)
            .title(config.window_title.as_str())
            .inner_size(config.width, config.height)
            .min_inner_size(960.0, 680.0)
            .resizable(true);

    if let Some(icon) = app.default_window_icon() {
        window_builder = window_builder.icon(icon.clone())?;
    }

    window_builder.build()?;
    Ok(())
}

pub fn handle_run_event<R: tauri::Runtime>(app_handle: &tauri::AppHandle<R>, event: &RunEvent) {
    if let RunEvent::WindowEvent { label, event, .. } = event {
        if label == "main" {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                app_handle.exit(0);
            }
        }
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
