use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::ServeMode;
use console::style;
use izwi_core::ServeRuntimeConfig;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub struct ServeArgs {
    pub mode: ServeMode,
    pub runtime: ServeRuntimeConfig,
    pub log_level: String,
    pub dev: bool,
}

pub async fn execute(args: ServeArgs) -> Result<()> {
    let theme = Theme::default();

    theme.print_banner();

    let platform = detect_platform();
    println!("   Platform: {}", style(&platform).cyan());

    println!("\n{}", style("Configuration:").bold().underlined());
    println!(
        "  Mode:           {}",
        style(serve_mode_label(&args.mode)).cyan()
    );
    println!(
        "  Host:           {}:{}",
        args.runtime.host, args.runtime.port
    );
    println!("  Models dir:     {}", args.runtime.models_dir.display());
    println!("  Max batch:      {}", args.runtime.max_batch_size);
    println!("  Max concurrent: {}", args.runtime.max_concurrent_requests);
    println!("  Timeout:        {}s", args.runtime.request_timeout_secs);
    println!("  Backend:        {}", args.runtime.backend.as_str());
    println!("  Log level:      {}", args.log_level);

    set_server_env(&args);

    println!("\n{}", style("Starting server...").bold());
    let mut server_child = spawn_server(&args)?;

    let connect_host = server_connect_host(&args.runtime.host);
    let api_endpoint = format!("http://{}:{}/v1", connect_host, args.runtime.port);
    let web_ui = format!("http://{}:{}", connect_host, args.runtime.port);
    let browser_target = browser_target(&connect_host, args.runtime.port, !args.runtime.ui_enabled);

    match &args.mode {
        ServeMode::Server => {
            println!("\n{}", style("Server is running!").green().bold());
            println!("  API endpoint: {}", style(&api_endpoint).cyan());
            if args.runtime.ui_enabled {
                println!("  Web UI:       {}", style(&web_ui).cyan());
            }
            println!("\nPress Ctrl+C to stop the server.\n");

            let status = server_child
                .wait()
                .map_err(|e| CliError::Other(format!("Server error: {}", e)))?;

            if !status.success() {
                return Err(CliError::Other(format!(
                    "Server exited with code: {:?}",
                    status.code()
                )));
            }
        }
        ServeMode::Desktop => {
            if let Err(err) = wait_for_server_ready(&api_endpoint, Duration::from_secs(30)).await {
                let _ = shutdown_child(&mut server_child, "server");
                return Err(err);
            }

            println!("\n{}", style("Server is running!").green().bold());
            println!("  API endpoint: {}", style(&api_endpoint).cyan());
            println!("  Desktop URL:  {}", style(&web_ui).cyan());
            println!("  Launching desktop app...");

            let mut desktop_child = spawn_desktop(&args, &web_ui)?;
            println!("\n{}", style("Desktop app is running.").green().bold());
            println!("Close the desktop window or press Ctrl+C to stop.\n");

            supervise_desktop_mode(&mut server_child, &mut desktop_child).await?;
        }
        ServeMode::Web => {
            if let Err(err) = wait_for_server_ready(&api_endpoint, Duration::from_secs(30)).await {
                let _ = shutdown_child(&mut server_child, "server");
                return Err(err);
            }

            println!("\n{}", style("Server is running!").green().bold());
            println!("  API endpoint: {}", style(&api_endpoint).cyan());

            if !args.runtime.ui_enabled {
                eprintln!(
                    "{}",
                    style("Web mode requested with --no-ui; opening the health endpoint instead.")
                        .yellow()
                );
            }

            println!(
                "  {}:      {}",
                if args.runtime.ui_enabled {
                    "Web URL"
                } else {
                    "API URL"
                },
                style(&browser_target).cyan()
            );
            println!("  Launching browser...");

            if let Err(err) = open_in_browser(&browser_target) {
                eprintln!(
                    "{}",
                    style(format!(
                        "Could not launch browser automatically: {}. Open {} manually.",
                        err, browser_target
                    ))
                    .yellow()
                );
            } else {
                println!("{}", style("  Browser opened.").dim());
            }

            println!("\nPress Ctrl+C to stop the server.\n");

            let status = server_child
                .wait()
                .map_err(|e| CliError::Other(format!("Server error: {}", e)))?;

            if !status.success() {
                return Err(CliError::Other(format!(
                    "Server exited with code: {:?}",
                    status.code()
                )));
            }
        }
    }

    Ok(())
}

fn serve_mode_label(mode: &ServeMode) -> &'static str {
    match mode {
        ServeMode::Server => "server",
        ServeMode::Desktop => "desktop",
        ServeMode::Web => "web",
    }
}

fn set_server_env(args: &ServeArgs) {
    std::env::set_var("RUST_LOG", &args.log_level);
    std::env::set_var("IZWI_HOST", &args.runtime.host);
    std::env::set_var("IZWI_PORT", args.runtime.port.to_string());
    std::env::set_var(
        "IZWI_MODELS_DIR",
        args.runtime.models_dir.to_string_lossy().to_string(),
    );
    std::env::set_var(
        "IZWI_MAX_BATCH_SIZE",
        args.runtime.max_batch_size.to_string(),
    );
    std::env::set_var("IZWI_BACKEND", args.runtime.backend.as_str());
    std::env::set_var("IZWI_NUM_THREADS", args.runtime.num_threads.to_string());
    std::env::set_var(
        "IZWI_MAX_CONCURRENT",
        args.runtime.max_concurrent_requests.to_string(),
    );
    std::env::set_var(
        "IZWI_TIMEOUT",
        args.runtime.request_timeout_secs.to_string(),
    );
    std::env::set_var(
        "IZWI_CORS",
        if args.runtime.cors_enabled { "1" } else { "0" },
    );
    std::env::set_var("IZWI_CORS_ORIGINS", args.runtime.cors_origins.join(","));
    std::env::set_var(
        "IZWI_NO_UI",
        if args.runtime.ui_enabled { "0" } else { "1" },
    );
    std::env::set_var(
        "IZWI_UI_DIR",
        args.runtime.ui_dir.to_string_lossy().to_string(),
    );
    std::env::set_var("IZWI_SERVE_MODE", serve_mode_label(&args.mode));
}

fn spawn_server(args: &ServeArgs) -> Result<Child> {
    let server_binary = if args.dev {
        "cargo".to_string()
    } else {
        let binary_name = platform_binary_name("izwi-server");
        let binary_path = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .map(|p| p.join(&binary_name))
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.join("target/release").join(&binary_name))
            })
            .unwrap_or_else(|| PathBuf::from(&binary_name));

        if binary_path.exists() {
            binary_path.to_string_lossy().to_string()
        } else {
            println!("  {}", style("Using development mode (cargo run)").yellow());
            "cargo".to_string()
        }
    };

    let mut cmd = if server_binary == "cargo" {
        let mut c = Command::new("cargo");
        c.arg("run").arg("--bin").arg("izwi-server");
        if !args.dev {
            c.arg("--release");
        }
        c
    } else {
        Command::new(server_binary)
    };

    cmd.env("RUST_LOG", &args.log_level);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    cmd.spawn()
        .map_err(|e| CliError::Other(format!("Failed to start server: {}", e)))
}

fn spawn_desktop(args: &ServeArgs, server_url: &str) -> Result<Child> {
    #[cfg(target_os = "macos")]
    if !args.dev {
        if let Some(app_bundle) = resolve_macos_desktop_bundle() {
            println!(
                "  {}",
                style(format!("Using app bundle {}", app_bundle.display())).dim()
            );
            let mut cmd = Command::new("open");
            cmd.arg("-W")
                .arg("-n")
                .arg(&app_bundle)
                .arg("--args")
                .arg("--server-url")
                .arg(server_url)
                .arg("--window-title")
                .arg("Izwi");

            cmd.stdout(Stdio::inherit());
            cmd.stderr(Stdio::inherit());

            return cmd
                .spawn()
                .map_err(|e| CliError::Other(format!("Failed to start desktop app: {}", e)));
        }
    }

    let desktop_binary = if args.dev {
        "cargo".to_string()
    } else {
        let binary_name = platform_binary_name("izwi-desktop");
        let binary_path = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .map(|p| p.join(&binary_name))
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.join("target/release").join(&binary_name))
            })
            .unwrap_or_else(|| PathBuf::from(&binary_name));

        if binary_path.exists() {
            binary_path.to_string_lossy().to_string()
        } else {
            println!(
                "  {}",
                style("Desktop binary not found, using cargo run fallback").yellow()
            );
            "cargo".to_string()
        }
    };

    let mut cmd = if desktop_binary == "cargo" {
        let mut c = Command::new("cargo");
        c.arg("run").arg("--bin").arg("izwi-desktop");
        if !args.dev {
            c.arg("--release");
        }
        c.arg("--")
            .arg("--server-url")
            .arg(server_url)
            .arg("--window-title")
            .arg("Izwi");
        c
    } else {
        let mut c = Command::new(desktop_binary);
        c.arg("--server-url")
            .arg(server_url)
            .arg("--window-title")
            .arg("Izwi");
        c
    };

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    cmd.spawn()
        .map_err(|e| CliError::Other(format!("Failed to start desktop app: {}", e)))
}

#[cfg(target_os = "macos")]
fn resolve_macos_desktop_bundle() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("IZWI_DESKTOP_APP") {
        let candidate = PathBuf::from(path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(bundle) = find_macos_bundle_ancestor(&exe) {
            return Some(bundle);
        }

        if let Some(parent) = exe.parent() {
            let sibling_bundle = parent.join("Izwi.app");
            if sibling_bundle.exists() {
                return Some(sibling_bundle);
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let local_bundle = cwd
            .join("target")
            .join("release")
            .join("bundle")
            .join("macos")
            .join("Izwi.app");
        if local_bundle.exists() {
            return Some(local_bundle);
        }
    }

    let applications_bundle = PathBuf::from("/Applications/Izwi.app");
    if applications_bundle.exists() {
        Some(applications_bundle)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
fn find_macos_bundle_ancestor(path: &std::path::Path) -> Option<PathBuf> {
    path.ancestors()
        .find(|ancestor| ancestor.extension().and_then(|ext| ext.to_str()) == Some("app"))
        .map(|ancestor| ancestor.to_path_buf())
}

fn open_in_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut cmd = {
        let mut c = Command::new("open");
        c.arg(url);
        c
    };

    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", "", url]);
        c
    };

    #[cfg(all(unix, not(target_os = "macos")))]
    let mut cmd = {
        let mut c = Command::new("xdg-open");
        c.arg(url);
        c
    };

    cmd.stdout(Stdio::null());
    cmd.stderr(Stdio::null());
    cmd.spawn()
        .map_err(|e| CliError::Other(format!("Failed to launch browser: {}", e)))?;
    Ok(())
}

async fn wait_for_server_ready(api_endpoint: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()?;

    let health_url = format!("{}/health", api_endpoint);
    let deadline = Instant::now() + timeout;

    loop {
        if let Ok(resp) = client.get(&health_url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }

        if Instant::now() >= deadline {
            return Err(CliError::Other(format!(
                "Server did not become ready within {}s ({})",
                timeout.as_secs(),
                health_url
            )));
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

async fn supervise_desktop_mode(server: &mut Child, desktop: &mut Child) -> Result<()> {
    loop {
        if let Some(status) = server.try_wait()? {
            let _ = shutdown_child(desktop, "desktop app");
            return Err(CliError::Other(format!(
                "Server exited while desktop app was running (code: {:?})",
                status.code()
            )));
        }

        if let Some(status) = desktop.try_wait()? {
            if !status.success() {
                eprintln!(
                    "{}",
                    style(format!(
                        "Desktop app exited with code {:?}; shutting down server.",
                        status.code()
                    ))
                    .yellow()
                );
            }
            shutdown_child(server, "server")?;
            return Ok(());
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn shutdown_child(child: &mut Child, name: &str) -> Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }

    request_graceful_termination(child);

    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(8);
    const SHUTDOWN_POLL: Duration = Duration::from_millis(100);
    let start = Instant::now();
    while start.elapsed() < SHUTDOWN_TIMEOUT {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        thread::sleep(SHUTDOWN_POLL);
    }

    child
        .kill()
        .map_err(|e| CliError::Other(format!("Failed to stop {}: {}", name, e)))?;

    child
        .wait()
        .map_err(|e| CliError::Other(format!("Failed while waiting for {}: {}", name, e)))?;

    Ok(())
}

fn request_graceful_termination(child: &Child) {
    #[cfg(unix)]
    {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    #[cfg(not(unix))]
    let _ = child;
}

fn platform_binary_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{}.exe", base)
    } else {
        base.to_string()
    }
}

fn server_connect_host(host: &str) -> String {
    match host {
        "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        other => other.to_string(),
    }
}

fn browser_target(host: &str, port: u16, no_ui: bool) -> String {
    if no_ui {
        format!("http://{}:{}/v1/health", host, port)
    } else {
        format!("http://{}:{}", host, port)
    }
}

fn detect_platform() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let mut features = vec![];

    if cfg!(target_os = "macos") {
        features.push("Metal");
    }

    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        features.push("CUDA");
    }

    let feature_str = if features.is_empty() {
        String::new()
    } else {
        format!(" [{}]", features.join(", "))
    };

    format!("{}-{}{}", os, arch, feature_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::backends::BackendPreference;

    fn clear_serve_env() {
        std::env::remove_var("RUST_LOG");
        std::env::remove_var("IZWI_HOST");
        std::env::remove_var("IZWI_PORT");
        std::env::remove_var("IZWI_MAX_BATCH_SIZE");
        std::env::remove_var("IZWI_MAX_CONCURRENT");
        std::env::remove_var("IZWI_TIMEOUT");
        std::env::remove_var("IZWI_SERVE_MODE");
        std::env::remove_var("IZWI_BACKEND");
        std::env::remove_var("IZWI_NUM_THREADS");
        std::env::remove_var("IZWI_MODELS_DIR");
        std::env::remove_var("IZWI_CORS");
        std::env::remove_var("IZWI_CORS_ORIGINS");
        std::env::remove_var("IZWI_NO_UI");
        std::env::remove_var("IZWI_UI_DIR");
    }

    fn sample_args() -> ServeArgs {
        ServeArgs {
            mode: ServeMode::Web,
            runtime: ServeRuntimeConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                models_dir: PathBuf::from("/tmp/models"),
                max_batch_size: 8,
                backend: BackendPreference::Auto,
                num_threads: 4,
                max_concurrent_requests: 100,
                request_timeout_secs: 300,
                cors_enabled: true,
                cors_origins: vec!["*".to_string()],
                ui_enabled: false,
                ui_dir: PathBuf::from("/tmp/ui"),
            },
            log_level: "info".to_string(),
            dev: false,
        }
    }

    #[test]
    fn set_server_env_sets_ui_and_cors_flags() {
        clear_serve_env();

        set_server_env(&sample_args());

        assert_eq!(std::env::var("IZWI_CORS").as_deref(), Ok("1"));
        assert_eq!(std::env::var("IZWI_NO_UI").as_deref(), Ok("1"));
        assert_eq!(
            std::env::var("IZWI_MODELS_DIR").as_deref(),
            Ok("/tmp/models")
        );
        clear_serve_env();
    }

    #[test]
    fn browser_target_uses_health_when_ui_is_disabled() {
        assert_eq!(
            browser_target("127.0.0.1", 8080, true),
            "http://127.0.0.1:8080/v1/health"
        );
        assert_eq!(
            browser_target("127.0.0.1", 8080, false),
            "http://127.0.0.1:8080"
        );
    }
}
