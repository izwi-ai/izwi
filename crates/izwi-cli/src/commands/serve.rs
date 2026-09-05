use crate::error::{CliError, Result};
use crate::style::Theme;
use crate::{LogFormat, ServeMode};
use console::style;
use izwi_core::ServeRuntimeConfig;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub struct ServeArgs {
    pub config_path: Option<PathBuf>,
    pub mode: ServeMode,
    pub runtime: ServeRuntimeConfig,
    pub log_level: String,
    pub log_format: LogFormat,
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
    println!("  Tensor batch:   {}", args.runtime.max_batch_size);
    let physical_capacity = args
        .runtime
        .engine_config()
        .resolved_physical_execution_capacity();
    println!(
        "  Physical exec:  {} (configured {}, effective {})",
        args.runtime.physical_execution_mode,
        args.runtime.max_physical_in_flight,
        physical_capacity.physical_launch_limit
    );
    println!(
        "  Scheduler rows: {}",
        args.runtime.max_scheduler_batch_size
    );
    println!("  Retained seqs:  {}", args.runtime.max_retained_sequences);
    println!("  Staged txns:    {}", args.runtime.max_staged_transactions);
    println!("  Runtime queue:  {}", args.runtime.max_queued_requests);
    println!("  Context:        {}", args.runtime.max_sequence_length);
    println!("  Max concurrent: {}", args.runtime.max_concurrent_requests);
    println!("  Timeout:        {}s", args.runtime.request_timeout_secs);
    println!("  Backend:        {}", args.runtime.backend.as_str());
    println!("  Log level:      {}", args.log_level);
    println!("  Log format:     {}", args.log_format.as_str());

    args.runtime
        .performance
        .validate()
        .map_err(|error| CliError::ConfigError(error.to_string()))?;

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
                    style(
                        "Web mode requested with --no-ui; opening the readiness endpoint instead.",
                    )
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

fn configure_server_command(cmd: &mut Command, args: &ServeArgs) -> Result<()> {
    if let Some(path) = &args.config_path {
        cmd.arg("--config").arg(path);
    }
    cmd.env("RUST_LOG", &args.log_level);
    cmd.env("IZWI_LOG_FORMAT", args.log_format.as_str());
    cmd.env("IZWI_HOST", &args.runtime.host);
    cmd.env("IZWI_PORT", args.runtime.port.to_string());
    cmd.env(
        "IZWI_MODELS_DIR",
        args.runtime.models_dir.to_string_lossy().to_string(),
    );
    cmd.env(
        "IZWI_MAX_BATCH_SIZE",
        args.runtime.max_batch_size.to_string(),
    );
    cmd.env(
        "IZWI_PHYSICAL_EXECUTION_MODE",
        args.runtime.physical_execution_mode.to_string(),
    );
    cmd.env(
        "IZWI_MAX_PHYSICAL_IN_FLIGHT",
        args.runtime.max_physical_in_flight.to_string(),
    );
    cmd.env(
        "IZWI_MAX_SCHEDULER_BATCH_SIZE",
        args.runtime.max_scheduler_batch_size.to_string(),
    );
    cmd.env(
        "IZWI_MAX_LOADED_MODELS",
        args.runtime.max_loaded_models.to_string(),
    );
    cmd.env(
        "IZWI_MAX_RETAINED_SEQUENCES",
        args.runtime.max_retained_sequences.to_string(),
    );
    cmd.env(
        "IZWI_MAX_STAGED_TRANSACTIONS",
        args.runtime.max_staged_transactions.to_string(),
    );
    cmd.env(
        "IZWI_MAX_QUEUED_REQUESTS",
        args.runtime.max_queued_requests.to_string(),
    );
    cmd.env(
        "IZWI_MAX_SEQUENCE_LENGTH",
        args.runtime.max_sequence_length.to_string(),
    );
    cmd.env("IZWI_BACKEND", args.runtime.backend.as_str());
    cmd.env("IZWI_NUM_THREADS", args.runtime.num_threads.to_string());
    cmd.env(
        "IZWI_MAX_CONCURRENT",
        args.runtime.max_concurrent_requests.to_string(),
    );
    cmd.env(
        "IZWI_TIMEOUT",
        args.runtime.request_timeout_secs.to_string(),
    );
    cmd.env(
        "IZWI_CORS",
        if args.runtime.cors_enabled { "1" } else { "0" },
    );
    cmd.env("IZWI_CORS_ORIGINS", args.runtime.cors_origins.join(","));
    cmd.env(
        "IZWI_NO_UI",
        if args.runtime.ui_enabled { "0" } else { "1" },
    );
    cmd.env(
        "IZWI_UI_DIR",
        args.runtime.ui_dir.to_string_lossy().to_string(),
    );
    cmd.env("IZWI_SERVE_MODE", serve_mode_label(&args.mode));

    cmd.env(
        "IZWI_ENABLE_PREFIX_CACHING",
        args.runtime.enable_prefix_caching.to_string(),
    );
    cmd.env(
        "IZWI_ENABLE_CHUNKED_PREFILL",
        args.runtime.enable_chunked_prefill.to_string(),
    );
    cmd.env(
        "IZWI_MAX_PREFIX_CACHE_PAGES",
        args.runtime.max_prefix_cache_pages.to_string(),
    );
    cmd.env(
        "IZWI_CHUNKED_PREFILL_THRESHOLD",
        args.runtime.chunked_prefill_threshold.to_string(),
    );
    if let Some(salt) = &args.runtime.managed_prefix_cache_salt {
        cmd.env("IZWI_MANAGED_PREFIX_CACHE_SALT", salt);
    } else {
        cmd.env_remove("IZWI_MANAGED_PREFIX_CACHE_SALT");
    }
    // Serialize the complete resolved policy onto this child, so inherited
    // legacy switches cannot override a higher-precedence CLI value. Remove
    // aliases only in the child; the parent environment remains untouched.
    args.runtime
        .performance
        .validate()
        .map_err(|error| CliError::ConfigError(error.to_string()))?;
    let performance = serde_json::to_value(&args.runtime.performance)
        .map_err(|error| CliError::ConfigError(error.to_string()))?;
    for binding in izwi_core::performance::ENVIRONMENT_BINDINGS {
        for alias in binding.aliases {
            cmd.env_remove(alias);
        }
        let (group, field) = binding.key.split_once('.').expect("static performance key");
        let value = &performance[group][field];
        if value.is_null() {
            cmd.env_remove(binding.canonical);
        } else if let Some(value) = value.as_str() {
            cmd.env(binding.canonical, value);
        } else {
            cmd.env(binding.canonical, value.to_string());
        }
    }
    Ok(())
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
        c.arg("--");
        c
    } else {
        Command::new(server_binary)
    };

    configure_server_command(&mut cmd, args)?;
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

    let ready_url = readiness_url(api_endpoint);
    let deadline = Instant::now() + timeout;

    loop {
        if let Ok(resp) = client.get(&ready_url).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }

        if Instant::now() >= deadline {
            return Err(CliError::Other(format!(
                "Server did not become ready within {}s ({})",
                timeout.as_secs(),
                ready_url
            )));
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn readiness_url(api_endpoint: &str) -> String {
    format!("{}/ready", api_endpoint)
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
        format!("http://{}:{}/v1/ready", host, port)
    } else {
        format!("http://{}:{}", host, port)
    }
}

fn detect_platform() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let mut backends = vec!["CPU"];

    if cfg!(feature = "metal") {
        backends.push("Metal");
    }
    if cfg!(feature = "cuda") {
        backends.push("CUDA");
    }

    let feature_str = if backends.is_empty() {
        String::new()
    } else {
        format!(" [{}]", backends.join(", "))
    };

    format!("{}-{}{}", os, arch, feature_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::backends::BackendPreference;

    fn command_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.get_envs()
            .find(|(name, _)| *name == key)
            .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }

    fn sample_args() -> ServeArgs {
        ServeArgs {
            config_path: None,
            mode: ServeMode::Web,
            runtime: ServeRuntimeConfig {
                performance: Default::default(),
                host: "0.0.0.0".to_string(),
                port: 8080,
                models_dir: PathBuf::from("/tmp/models"),
                max_loaded_models: 1,
                max_batch_size: izwi_core::BatchSizePreference::Auto,
                physical_execution_mode: izwi_core::PhysicalExecutionMode::Shadow,
                max_physical_in_flight: izwi_core::PhysicalInFlightLimit::new(3).unwrap(),
                max_scheduler_batch_size: 8,
                enable_prefix_caching: false,
                managed_prefix_cache_salt: None,
                max_prefix_cache_pages: 128,
                enable_chunked_prefill: false,
                chunked_prefill_threshold: 192,
                max_retained_sequences: 8,
                max_staged_transactions: 8,
                max_queued_requests: 128,
                max_sequence_length: izwi_core::ContextLengthPreference::Auto,
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
            log_format: LogFormat::Text,
            dev: false,
        }
    }

    #[test]
    fn server_command_sets_ui_and_cors_flags() {
        let mut cmd = Command::new("unused-server");

        configure_server_command(&mut cmd, &sample_args()).unwrap();

        assert_eq!(command_env(&cmd, "IZWI_CORS").as_deref(), Some("1"));
        assert_eq!(command_env(&cmd, "IZWI_NO_UI").as_deref(), Some("1"));
        assert_eq!(
            command_env(&cmd, "IZWI_LOG_FORMAT").as_deref(),
            Some("text")
        );
        assert_eq!(
            command_env(&cmd, "IZWI_PHYSICAL_EXECUTION_MODE").as_deref(),
            Some("shadow")
        );
        assert_eq!(
            command_env(&cmd, "IZWI_MAX_PHYSICAL_IN_FLIGHT").as_deref(),
            Some("3")
        );
        assert_eq!(
            command_env(&cmd, "IZWI_MODELS_DIR").as_deref(),
            Some("/tmp/models")
        );
    }

    #[test]
    fn server_command_passes_json_log_format() {
        let mut cmd = Command::new("unused-server");

        let mut args = sample_args();
        args.log_format = LogFormat::Json;
        configure_server_command(&mut cmd, &args).unwrap();

        assert_eq!(
            command_env(&cmd, "IZWI_LOG_FORMAT").as_deref(),
            Some("json")
        );
    }

    #[test]
    fn browser_target_uses_readiness_when_ui_is_disabled() {
        assert_eq!(
            browser_target("127.0.0.1", 8080, true),
            "http://127.0.0.1:8080/v1/ready"
        );
        assert_eq!(
            browser_target("127.0.0.1", 8080, false),
            "http://127.0.0.1:8080"
        );
    }

    #[test]
    fn startup_wait_uses_readiness_endpoint() {
        assert_eq!(
            readiness_url("http://127.0.0.1:8080/v1"),
            "http://127.0.0.1:8080/v1/ready"
        );
    }

    #[test]
    fn server_child_receives_resolved_performance_without_parent_env_mutation() {
        let _guard = crate::test_support::env_lock();
        let mut args = sample_args();
        args.runtime.performance.cuda.mode = izwi_core::OptimizationMode::Off;
        args.runtime.performance.cuda.mtp_adaptive = false;
        args.runtime.performance.loading.workers = 0;
        args.runtime.performance.loading.cache_dir = Some(PathBuf::from("/tmp/child cache"));
        let before: std::collections::BTreeMap<_, _> = std::env::vars_os().collect();
        let mut cmd = Command::new("unused-server");
        cmd.env("IZWI_QWEN38_MTP", "1");
        cmd.env("IZWI_CUDA_MTP_ADAPTIVE", "true");
        configure_server_command(&mut cmd, &args).unwrap();
        let after: std::collections::BTreeMap<_, _> = std::env::vars_os().collect();
        assert_eq!(before, after);
        assert_eq!(command_env(&cmd, "IZWI_CUDA_MODE").as_deref(), Some("off"));
        assert_eq!(
            command_env(&cmd, "IZWI_CUDA_MTP_ADAPTIVE").as_deref(),
            Some("false")
        );
        assert_eq!(
            command_env(&cmd, "IZWI_LOADING_WORKERS").as_deref(),
            Some("0")
        );
        assert_eq!(
            command_env(&cmd, "IZWI_LOADING_CACHE_DIR").as_deref(),
            Some("/tmp/child cache")
        );
        assert!(cmd
            .get_envs()
            .any(|(name, value)| name == "IZWI_QWEN38_MTP" && value.is_none()));
        let child_overrides =
            izwi_core::PerformanceConfigOverrides::from_lookup(|key| command_env(&cmd, key))
                .unwrap();
        let mut child = izwi_core::PerformanceConfig::default();
        child.apply_overrides(&child_overrides);
        assert_eq!(child.cuda, args.runtime.performance.cuda);
        assert_eq!(child.loading, args.runtime.performance.loading);
        assert!(!child.normalized().cuda.mtp.enabled());
    }

    #[test]
    fn performance_startup_passes_the_selected_config_file_to_server_child() {
        let mut args = sample_args();
        args.config_path = Some(PathBuf::from("/tmp/selected config.toml"));
        let mut command = Command::new("unused-server");
        configure_server_command(&mut command, &args).unwrap();
        let arguments: Vec<_> = command.get_args().collect();
        assert_eq!(
            arguments,
            [
                std::ffi::OsStr::new("--config"),
                std::ffi::OsStr::new("/tmp/selected config.toml")
            ]
        );
    }
}
