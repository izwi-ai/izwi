use anyhow::{Context, Result};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;
use url::Url;

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
) -> Result<Option<Child>> {
    const START_TIMEOUT: Duration = Duration::from_secs(15);
    const POLL_INTERVAL: Duration = Duration::from_millis(200);
    const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);

    let (host, port) = server_host_port(server_url)?;
    if !is_local_server_host(&host) {
        return Ok(None);
    }

    if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
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

    cmd.env("IZWI_HOST", bind_host)
        .env("IZWI_PORT", port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .stdin(Stdio::null());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to start izwi-server for {}:{}", host, port))?;

    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
            return Ok(Some(child));
        }

        if let Some(status) = child
            .try_wait()
            .context("failed while checking izwi-server status")?
        {
            anyhow::bail!(
                "izwi-server exited before becoming ready on {}:{} (status: {})",
                host,
                port,
                status
            );
        }

        thread::sleep(POLL_INTERVAL);
    }

    shutdown_child(&mut child);
    anyhow::bail!("timed out waiting for izwi-server on {}:{}", host, port)
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
}
