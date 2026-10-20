use anyhow::{Context, Result};
use clap::Parser;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use url::Url;

pub mod alerts;
pub mod autostart;
pub mod blocking;
pub mod console;
pub mod csp;
pub mod desktop_log;
pub mod downloads;
pub mod install;
pub mod reveal;
pub mod server;
pub mod settings;
pub mod stale_server;
pub mod tray;
pub mod tray_visibility;
pub mod updater;
pub mod updater_contract;
pub mod window;
pub mod window_state;

use self::server::{server_host_port, start_local_server, ManagedServer};
use self::tray_visibility::TrayVisibilityState;
use self::window::WindowConfig;
use tauri::Manager;

const DEFAULT_SERVER_URL: &str = "http://localhost:8080";

#[derive(Debug, Parser)]
#[command(
    name = "izwi-desktop",
    about = "Tauri desktop shell for Izwi local inference",
    version
)]
pub struct DesktopArgs {
    /// Base URL of the Izwi local server [default: http://localhost:8080].
    /// Without it, the app starts its server on another port when another
    /// program is using 8080.
    #[arg(long, value_name = "URL")]
    pub server_url: Option<String>,

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
    desktop_log::install_panic_hook();
    // An explicit URL names the intended server; only the default may move.
    let may_change_port = args.server_url.is_none();
    let server_url_arg = args.server_url.as_deref().unwrap_or(DEFAULT_SERVER_URL);
    let server_url = Url::parse(server_url_arg)
        .with_context(|| format!("invalid --server-url value: {server_url_arg}"))?;
    server_host_port(&server_url)?;
    let window_title = args.window_title;
    let (width, height) = (args.width, args.height);

    let managed_server = Arc::new(Mutex::new(None::<ManagedServer>));
    let server_shutdown = Arc::new(AtomicBool::new(false));
    let setup_server_handle = Arc::clone(&managed_server);
    let setup_shutdown_handle = Arc::clone(&server_shutdown);
    let event_server_handle = Arc::clone(&managed_server);
    let event_shutdown_handle = Arc::clone(&server_shutdown);

    // Must be the first plugin: a second launch hands its arguments to this
    // instance and exits before it can spawn a second server, window or tray.
    let mut builder = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            let _ = tray::show_main_window(app);
        }))
        .invoke_handler(tauri::generate_handler![
            downloads::download_audio_file,
            downloads::save_file,
            reveal::reveal_path,
            autostart::launch_at_login_enabled,
            autostart::set_launch_at_login_enabled,
            tray_visibility::tray_icon_visible,
            tray_visibility::set_tray_icon_visible,
            updater::check_for_beta_update,
            updater::install_beta_update,
            updater::relaunch_after_update,
            updater::updater_health_snapshot,
            install::cli_install_status,
            install::install_cli,
        ])
        .manage(updater::UpdaterState::new())
        .manage(ServerControl {
            server: Arc::clone(&managed_server),
            shutdown: Arc::clone(&server_shutdown),
        })
        .manage(tray_visibility::TrayVisibilityState::new(true));

    if let Some(app_key) = resolve_aptabase_app_key() {
        builder = builder.plugin(tauri_plugin_aptabase::Builder::new(&app_key).build());
    }

    if let Some(pubkey) = updater::configured_pubkey() {
        builder = builder.plugin(tauri_plugin_updater::Builder::new().pubkey(pubkey).build());
    } else {
        desktop_log::warn("updater pubkey is not configured; in-app update checks are disabled");
    }

    builder = builder.plugin(tauri_plugin_autostart::Builder::new().build());
    builder = builder.plugin(tauri_plugin_dialog::init());
    builder = builder.plugin(tauri_plugin_process::init());

    let mut context = tauri::generate_context!();
    csp::allow_server_origin(context.config_mut(), &server_url);

    let app = builder
        .setup(move |app| {
            match app.path().app_log_dir() {
                Ok(dir) => match desktop_log::init(&dir) {
                    Ok(path) => desktop_log::info(format!(
                        "izwi-desktop {} starting; log: {}",
                        env!("CARGO_PKG_VERSION"),
                        path.display()
                    )),
                    Err(error) => desktop_log::warn(format!(
                        "could not open the desktop log in {}: {error}",
                        dir.display()
                    )),
                },
                Err(error) => {
                    desktop_log::warn(format!("could not resolve the log directory: {error}"))
                }
            }

            // Failures here are shown in a dialog that quits once read; an
            // error returned from setup would panic without a word.
            let local = match start_local_server(app.handle(), &server_url, may_change_port) {
                Ok(local) => local,
                Err(error) => {
                    desktop_log::error(format!("startup failed: {error:#}"));
                    alerts::startup_failed(app.handle(), &error);
                    return Ok(());
                }
            };
            app.manage(server::ServerEndpoint::new(local.url.clone()));
            let window_config = WindowConfig {
                server_origin: server_origin(&local.url)?,
                window_title,
                width,
                height,
            };
            if let Some(server_child) = local.managed {
                desktop_log::info(format!(
                    "desktop-owned izwi-server at {}; logs: {}",
                    local.url,
                    server_child.log_path().display()
                ));
                {
                    let mut child_slot = setup_server_handle
                        .lock()
                        .map_err(|_| anyhow::anyhow!("failed to acquire server startup lock"))?;
                    *child_slot = Some(server_child);
                }
                server::spawn_server_monitor(
                    app.handle().clone(),
                    local.url.clone(),
                    Arc::clone(&setup_server_handle),
                    Arc::clone(&setup_shutdown_handle),
                );
            }

            // Installing the CLI is an explicit Settings action (it can need an
            // administrator password); launch only refreshes existing copies.
            if let Err(err) =
                install::ensure_cli_setup(app.handle(), install::CliSetupMode::RefreshExisting)
            {
                desktop_log::warn(format!("could not refresh terminal commands: {err:#}"));
            }

            if let Err(error) = window::build_main_window(app, &window_config) {
                desktop_log::error(format!("could not open the window: {error:#}"));
                alerts::startup_failed(
                    app.handle(),
                    &error.context("Izwi couldn't open its window."),
                );
                return Ok(());
            }

            let tray_shown = tray::desktop_shows_tray_icons()
                && match tray::build_basic_tray(app.handle()) {
                    Ok(()) => true,
                    Err(error) => {
                        desktop_log::warn(format!("no tray icon: {error:#}"));
                        false
                    }
                };
            if !tray_shown {
                desktop_log::info("no tray icon on this desktop; closing the window quits Izwi");
                app.state::<TrayVisibilityState>().set_available(false);
            }
            tray_visibility::restore_tray_visibility(app.handle());
            Ok(())
        })
        .build(context)
        .map_err(|e| anyhow::anyhow!("failed to build desktop app: {}", e))?;

    let exit_code = app.run_return(move |app_handle, event| {
        window::handle_run_event(app_handle, &event);
        if matches!(event, tauri::RunEvent::Exit) {
            // Wait here, the loop's last event: a restart after an update
            // follows it and must not race the server it replaces.
            stop_managed_server(&event_server_handle, &event_shutdown_handle);
        } else if window::event_ends_desktop_session(app_handle, &event) {
            // Only ask: waiting would hold up the event loop.
            request_server_stop(&event_server_handle, &event_shutdown_handle);
        }
    });

    stop_managed_server(&managed_server, &server_shutdown);

    if exit_code != 0 {
        return Err(anyhow::anyhow!(
            "desktop app exited with code {}",
            exit_code
        ));
    }

    Ok(())
}

/// Set the shutdown flag (so the monitor won't restart the server) and ask
/// the server to stop, without waiting.
fn request_server_stop(server: &Arc<Mutex<Option<ManagedServer>>>, shutdown: &Arc<AtomicBool>) {
    shutdown.store(true, Ordering::Release);
    if let Ok(mut server_slot) = server.lock() {
        if let Some(server) = server_slot.as_mut() {
            server.request_stop();
        }
    }
}

/// Stops the managed server from outside the event loop: before an update's
/// installer takes the app down on Windows.
pub struct ServerControl {
    server: Arc<Mutex<Option<ManagedServer>>>,
    shutdown: Arc<AtomicBool>,
}

impl ServerControl {
    pub fn stop(&self) {
        stop_managed_server(&self.server, &self.shutdown);
    }
}

fn stop_managed_server(server: &Arc<Mutex<Option<ManagedServer>>>, shutdown: &Arc<AtomicBool>) {
    shutdown.store(true, Ordering::Release);
    let taken = server
        .lock()
        .ok()
        .and_then(|mut server_slot| server_slot.take());
    // Outside the lock: this waits for the server to exit.
    if let Some(mut server) = taken {
        server.shutdown();
    }
}

fn server_origin(server_url: &Url) -> Result<String> {
    let (host, port) = server_host_port(server_url)?;
    Ok(format!("{}://{}:{}", server_url.scheme(), host, port))
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
