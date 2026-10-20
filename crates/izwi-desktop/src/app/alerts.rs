//! Native dialogs for failures the user has to know about: without them the
//! app would vanish, or sit there with a server that is gone.

use tauri::{AppHandle, Manager, Runtime};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

/// Say why Izwi couldn't start, then quit once the user has read it.
pub fn startup_failed<R: Runtime>(app: &AppHandle<R>, error: &anyhow::Error) {
    let handle = app.clone();
    app.dialog()
        .message(with_logs_hint(app, &error.to_string()))
        .title("Izwi couldn't start")
        .kind(MessageDialogKind::Error)
        .show(move |_| handle.exit(1));
}

/// Say that the server stopped and restarting it failed.
pub fn server_stopped<R: Runtime>(app: &AppHandle<R>) {
    app.dialog()
        .message(with_logs_hint(
            app,
            "Izwi's server stopped and couldn't be restarted. Quit Izwi and open it again.",
        ))
        .title("Izwi's server stopped")
        .kind(MessageDialogKind::Error)
        .show(|_| {});
}

fn with_logs_hint<R: Runtime>(app: &AppHandle<R>, message: &str) -> String {
    match app.path().app_log_dir() {
        Ok(dir) => format!("{message}\n\nDetails are in the logs at {}.", dir.display()),
        Err(_) => message.to_string(),
    }
}
