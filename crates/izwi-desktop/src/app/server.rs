use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;
use url::Url;

const SERVER_LOG_FILE: &str = "izwi-server.log";
const DESKTOP_OWNER_PIPE_ENV: &str = "IZWI_DESKTOP_OWNER_PIPE";

pub struct ManagedServer {
    child: Option<Child>,
    log_path: PathBuf,
}

impl ManagedServer {
    fn new(child: Child, log_path: PathBuf) -> Self {
        Self {
            child: Some(child),
            log_path,
        }
    }

    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            shutdown_child(&mut child);
        }
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }
}

impl Drop for ManagedServer {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
struct LiveResponse {
    status: String,
    version: String,
}

pub fn server_host_port(server_url: &Url) -> Result<(String, u16)> {
    let host = server_url
        .host_str()
        .context("--server-url must include a host")?
        .to_string();
    let port = server_url
        .port_or_known_default()
        .context("--server-url must include a port or use a known scheme")?;
    Ok((host, port))
}

pub fn maybe_start_local_server<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    server_url: &Url,
) -> Result<Option<ManagedServer>> {
    const START_TIMEOUT: Duration = Duration::from_secs(15);
    const POLL_INTERVAL: Duration = Duration::from_millis(200);
    const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);

    let (host, port) = server_host_port(server_url)?;
    if !is_local_server_host(&host) {
        return Ok(None);
    }

    if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
        validate_existing_server(server_url, CONNECT_TIMEOUT)?;
        eprintln!(
            "warning: using compatible local izwi-server at {}; it was not started by the desktop app and will not be stopped on exit",
            server_url
        );
        return Ok(None);
    }

    let mut cmd = match resolve_server_binary(app) {
        Some(path) => Command::new(path),
        None => Command::new(platform_binary_name("izwi-server")),
    };

    let bind_host = if host == "localhost" {
        "127.0.0.1"
    } else {
        host.as_str()
    };

    let log_path = open_server_log(app, &mut cmd)?;

    cmd.env("IZWI_HOST", bind_host)
        .env("IZWI_PORT", port.to_string())
        .env(DESKTOP_OWNER_PIPE_ENV, "1")
        .stdin(Stdio::piped());

    let mut child = cmd.spawn().with_context(|| {
        format!(
            "failed to start izwi-server for {}:{} (log: {})",
            host,
            port,
            log_path.display()
        )
    })?;

    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if is_server_reachable(&host, port, CONNECT_TIMEOUT)
            && probe_server(server_url, CONNECT_TIMEOUT).is_ok()
        {
            return Ok(Some(ManagedServer::new(child, log_path)));
        }

        if let Some(status) = child
            .try_wait()
            .context("failed while checking izwi-server status")?
        {
            anyhow::bail!(
                "izwi-server exited before becoming ready on {}:{} (status: {}; log: {})",
                host,
                port,
                status,
                log_path.display()
            );
        }

        thread::sleep(POLL_INTERVAL);
    }

    shutdown_child(&mut child);
    anyhow::bail!(
        "timed out waiting for izwi-server on {}:{} (log: {})",
        host,
        port,
        log_path.display()
    )
}

fn open_server_log<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    cmd: &mut Command,
) -> Result<PathBuf> {
    let log_dir = app
        .path()
        .app_log_dir()
        .context("failed to resolve desktop log directory")?;
    fs::create_dir_all(&log_dir).with_context(|| {
        format!(
            "failed to create desktop log directory {}",
            log_dir.display()
        )
    })?;

    let log_path = log_dir.join(SERVER_LOG_FILE);
    let mut stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open server log {}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .with_context(|| format!("failed to clone server log {}", log_path.display()))?;
    writeln!(
        stdout,
        "\n--- izwi-desktop starting izwi-server ({:?}) ---",
        std::time::SystemTime::now()
    )
    .with_context(|| format!("failed to write server log {}", log_path.display()))?;

    cmd.stdout(Stdio::from(stdout)).stderr(Stdio::from(stderr));
    Ok(log_path)
}

fn validate_existing_server(server_url: &Url, timeout: Duration) -> Result<()> {
    probe_server(server_url, timeout).with_context(|| {
        format!(
            "a process is already listening at {}, but it is not a compatible izwi-server; stop it or pass --server-url for the intended server",
            server_url
        )
    })
}

fn probe_server(server_url: &Url, timeout: Duration) -> Result<()> {
    let live_url = server_url
        .join("/livez")
        .context("failed to construct izwi-server liveness URL")?;
    let response = reqwest::blocking::Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build izwi-server probe client")?
        .get(live_url)
        .send()
        .context("liveness probe failed")?
        .error_for_status()
        .context("liveness probe returned an error status")?;
    let body = response
        .text()
        .context("failed to read liveness response")?;
    validate_liveness_body(&body, env!("CARGO_PKG_VERSION"))
}

fn validate_liveness_body(body: &str, expected_version: &str) -> Result<()> {
    let response: LiveResponse =
        serde_json::from_str(body).context("liveness response was not valid Izwi JSON")?;
    if response.status != "alive" {
        anyhow::bail!("unexpected liveness status {:?}", response.status);
    }
    if response.version != expected_version {
        anyhow::bail!(
            "server version {} does not match desktop version {}",
            response.version,
            expected_version
        );
    }
    Ok(())
}

pub fn is_local_server_host(host: &str) -> bool {
    matches!(host, "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | "::")
}

pub fn platform_binary_name(name: &str) -> String {
    if cfg!(windows) {
        format!("{}.exe", name)
    } else {
        name.to_string()
    }
}

pub fn shutdown_child(child: &mut Child) {
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(8);
    const SHUTDOWN_POLL: Duration = Duration::from_millis(100);

    if child.try_wait().ok().flatten().is_some() {
        return;
    }

    request_graceful_termination(child);

    let start = Instant::now();
    while start.elapsed() < SHUTDOWN_TIMEOUT {
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => thread::sleep(SHUTDOWN_POLL),
            Err(_) => break,
        }
    }

    let _ = child.kill();
    let _ = child.wait();
}

fn is_server_reachable(host: &str, port: u16, timeout: Duration) -> bool {
    let addrs = match (host, port).to_socket_addrs() {
        Ok(addrs) => addrs.collect::<Vec<_>>(),
        Err(_) => return false,
    };

    addrs
        .iter()
        .any(|addr| TcpStream::connect_timeout(addr, timeout).is_ok())
}

fn resolve_server_binary<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> Option<PathBuf> {
    let binary_name = platform_binary_name("izwi-server");
    let mut candidates = Vec::new();

    if let Ok(resource_dir) = app.path().resource_dir() {
        candidates.push(resource_dir.join("bin").join(&binary_name));
        candidates.push(resource_dir.join(&binary_name));
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join(&binary_name));
        }
    }

    candidates.into_iter().find(|candidate| candidate.exists())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_host_port_uses_known_default_port() {
        let url = Url::parse("http://localhost").expect("url");
        let (host, port) = server_host_port(&url).expect("host/port");
        assert_eq!(host, "localhost");
        assert_eq!(port, 80);
    }

    #[test]
    fn liveness_probe_accepts_matching_izwi_server() {
        validate_liveness_body(
            r#"{"status":"alive","version":"0.1.0-test","uptime_secs":3}"#,
            "0.1.0-test",
        )
        .expect("matching server");
    }

    #[test]
    fn liveness_probe_rejects_incompatible_server_version() {
        let error = validate_liveness_body(
            r#"{"status":"alive","version":"0.0.9","uptime_secs":3}"#,
            "0.1.0",
        )
        .expect_err("version mismatch");

        assert!(error
            .to_string()
            .contains("server version 0.0.9 does not match desktop version 0.1.0"));
    }

    #[test]
    fn liveness_probe_rejects_non_izwi_response() {
        let error =
            validate_liveness_body(r#"{"ok":true}"#, "0.1.0").expect_err("non-Izwi response");

        assert!(error
            .to_string()
            .contains("liveness response was not valid Izwi JSON"));
    }
}
