use anyhow::{Context, Result};
use clap::Parser;
use std::sync::{Arc, Mutex};
use url::Url;

pub mod autostart;
pub mod downloads;
pub mod install;
pub mod server;
pub mod tray;
pub mod tray_visibility;
pub mod updater;
pub mod updater_contract;
pub mod window;

use self::server::{maybe_start_local_server, server_host_port, ManagedServer};
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

    let managed_server = Arc::new(Mutex::new(None::<ManagedServer>));
    let setup_server_handle = Arc::clone(&managed_server);
    let event_server_handle = Arc::clone(&managed_server);

    let mut builder = tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            downloads::download_audio_file,
            autostart::launch_at_login_enabled,
            autostart::set_launch_at_login_enabled,
            tray_visibility::tray_icon_visible,
            tray_visibility::set_tray_icon_visible,
            updater::check_for_beta_update,
            updater::install_beta_update,
            updater::relaunch_after_update,
            updater::updater_health_snapshot,
        ])
        .manage(updater::UpdaterState::new())
        .manage(tray_visibility::TrayVisibilityState::new(true));

    if let Some(app_key) = resolve_aptabase_app_key() {
        builder = builder.plugin(tauri_plugin_aptabase::Builder::new(&app_key).build());
    }

    if let Some(pubkey) = resolve_updater_pubkey() {
        builder = builder.plugin(tauri_plugin_updater::Builder::new().pubkey(pubkey).build());
    } else {
        eprintln!("warning: updater pubkey is not configured; in-app update checks are disabled");
    }

    builder = builder.plugin(tauri_plugin_autostart::Builder::new().build());
    builder = builder.plugin(tauri_plugin_process::init());

    let app = builder
        .setup(move |app| {
            if let Some(server_child) = maybe_start_local_server(app.handle(), &server_url)? {
                eprintln!(
                    "desktop-owned izwi-server logs: {}",
                    server_child.log_path().display()
                );
                let mut child_slot = setup_server_handle
                    .lock()
                    .map_err(|_| anyhow::anyhow!("failed to acquire server startup lock"))?;
                *child_slot = Some(server_child);
            }

            if let Err(err) = install::ensure_cli_setup(app.handle()) {
                eprintln!("warning: could not configure terminal commands automatically: {err}");
            }

            window::build_main_window(app, &window_config)?;
            tray::build_basic_tray(app.handle())?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .map_err(|e| anyhow::anyhow!("failed to build desktop app: {}", e))?;

    let exit_code = app.run_return(move |app_handle, event| {
        window::handle_run_event(app_handle, &event);
        if window::event_ends_desktop_session(app_handle, &event) {
            stop_managed_server(&event_server_handle);
        }
    });

    stop_managed_server(&managed_server);

    if exit_code != 0 {
        return Err(anyhow::anyhow!(
            "desktop app exited with code {}",
            exit_code
        ));
    }

    Ok(())
}

fn stop_managed_server(server: &Arc<Mutex<Option<ManagedServer>>>) {
    if let Ok(mut server_slot) = server.lock() {
        if let Some(mut server) = server_slot.take() {
            server.shutdown();
        }
    }
}

fn resolve_aptabase_app_key() -> Option<String> {
    option_env!("APTABASE_APP_KEY")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("APTABASE_APP_KEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}

fn resolve_updater_pubkey() -> Option<String> {
    option_env!("IZWI_UPDATER_PUBKEY")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("IZWI_UPDATER_PUBKEY")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
}
