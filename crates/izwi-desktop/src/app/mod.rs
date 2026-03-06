use anyhow::{Context, Result};
use clap::Parser;
use std::process::Child;
use std::sync::{Arc, Mutex};
use url::Url;

pub mod downloads;
pub mod install;
pub mod server;
pub mod window;

use self::server::{maybe_start_local_server, server_host_port, shutdown_child};
use self::window::WindowConfig;

#[derive(Debug, Parser)]
#[command(
    name = "izwi-desktop",
    about = "Tauri desktop shell for Izwi local inference",
    version
)]
pub struct DesktopArgs {
    /// Base URL of the Izwi local server
    #[arg(long, default_value = "http://localhost:8080")]
    pub server_url: String,

    /// Desktop window title
    #[arg(long, default_value = "Izwi")]
    pub window_title: String,

    /// Initial window width
    #[arg(long, default_value = "1360")]
    pub width: f64,

    /// Initial window height
    #[arg(long, default_value = "900")]
    pub height: f64,
}

pub fn run(args: DesktopArgs) -> Result<()> {
    let server_url = Url::parse(&args.server_url)
        .with_context(|| format!("invalid --server-url value: {}", args.server_url))?;
    let (server_host, server_port) = server_host_port(&server_url)?;
    let server_origin = format!("{}://{}:{}", server_url.scheme(), server_host, server_port);
    let window_config = WindowConfig {
        server_origin,
        window_title: args.window_title,
        width: args.width,
        height: args.height,
    };

    let managed_server = Arc::new(Mutex::new(None::<Child>));
    let setup_server_handle = Arc::clone(&managed_server);

    let app = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![downloads::download_audio_file])
        .setup(move |app| {
            if let Some(server_child) = maybe_start_local_server(app.handle(), &server_url)? {
                let mut child_slot = setup_server_handle
                    .lock()
                    .map_err(|_| anyhow::anyhow!("failed to acquire server startup lock"))?;
                *child_slot = Some(server_child);
            }

            if let Err(err) = install::ensure_cli_setup(app.handle()) {
                eprintln!("warning: could not configure terminal commands automatically: {err}");
            }

            window::build_main_window(app, &window_config)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("failed to build desktop app: {}", e))?;

    let exit_code = app.run_return(|app_handle, event| {
        window::handle_run_event(app_handle, &event);
    });

    if let Ok(mut child_slot) = managed_server.lock() {
        if let Some(mut child) = child_slot.take() {
            shutdown_child(&mut child);
        }
    }

    if exit_code != 0 {
        return Err(anyhow::anyhow!(
            "desktop app exited with code {}",
            exit_code
        ));
    }

    Ok(())
}
