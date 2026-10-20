use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};
use tauri::Manager;
use url::Url;

use super::{alerts, desktop_log, stale_server};

const SERVER_LOG_FILE: &str = "izwi-server.log";
const DESKTOP_OWNER_PIPE_ENV: &str = "IZWI_DESKTOP_OWNER_PIPE";
const SHUTDOWN_GRACE_ENV: &str = "IZWI_HTTP_SHUTDOWN_GRACE_SECS";
/// Drain allowed to the desktop-owned server at quit, when nothing else sets
/// one. The server's 20 s default outlasts [`wait_or_kill`]'s 8 s whenever an
/// event stream is open, and a quitting app has no window left to answer.
const DESKTOP_SHUTDOWN_GRACE_SECS: &str = "3";
const CONNECT_TIMEOUT: Duration = Duration::from_millis(250);
/// How long whatever already listens on the port has to say what it is.
/// Generous: taking a busy Izwi server for another program would start a
/// second server on the same data.
const IDENTIFY_TIMEOUT: Duration = Duration::from_secs(3);
/// How long a new server may take to answer its liveness probe. A first
/// launch can be slow while the OS checks the binary.
const START_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a server from another Izwi version may take to go away. Right
/// after an update relaunches the app, the old version's server is still
/// shutting down.
const OTHER_IZWI_GRACE: Duration = Duration::from_secs(10);
/// How long a stale server we stopped may take to free the port.
const STALE_SERVER_STOP: Duration = Duration::from_secs(5);
/// Ports tried, in order, when the requested one is used by another program.
const FALLBACK_PORTS: RangeInclusive<u16> = 8081..=8099;
/// How often the monitor checks that the managed server is still running.
const MONITOR_TICK: Duration = Duration::from_secs(1);
/// How often the monitor polls `/livez` for a poisoned runtime.
const POISON_POLL_INTERVAL: Duration = Duration::from_secs(10);
const POISON_PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const RECREATION_MIN_COOLDOWN: Duration = Duration::from_secs(60);
const RECREATION_MAX_COOLDOWN: Duration = Duration::from_secs(300);
/// Pauses before each restart of a server that stopped unexpectedly, within
/// [`RESTART_WINDOW`]. One more failure and the monitor gives up.
const RESTART_DELAYS: [Duration; 3] = [
    Duration::from_secs(1),
    Duration::from_secs(5),
    Duration::from_secs(15),
];
const RESTART_WINDOW: Duration = Duration::from_secs(600);

/// Everything needed to (re)spawn the desktop-owned izwi-server. Kept on the
/// [`ManagedServer`] so a crashed or poisoned server can be recreated
/// identically.
#[derive(Debug, Clone)]
pub struct ServerSpawnSpec {
    binary: Option<PathBuf>,
    bind_host: String,
    port: u16,
}

pub struct ManagedServer {
    child: Option<Child>,
    log_path: PathBuf,
    spec: ServerSpawnSpec,
    stop_requested: bool,
}

impl ManagedServer {
    fn new(child: Child, log_path: PathBuf, spec: ServerSpawnSpec) -> Self {
        Self {
            child: Some(child),
            log_path,
            spec,
            stop_requested: false,
        }
    }

    /// Ask the server to stop without waiting for it; [`Self::shutdown`]
    /// waits. Quitting asks first so the event loop is not blocked while the
    /// server drains.
    pub fn request_stop(&mut self) {
        if self.stop_requested {
            return;
        }
        self.stop_requested = true;
        if let Some(child) = self.child.as_mut() {
            request_graceful_termination(child);
        }
    }

    /// Stop the server, killing it if it has not exited in time.
    pub fn shutdown(&mut self) {
        self.request_stop();
        if let Some(mut child) = self.child.take() {
            wait_or_kill(&mut child);
        }
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// The exit status once the process has exited.
    fn exit_status(&mut self) -> Option<ExitStatus> {
        self.child.as_mut()?.try_wait().ok().flatten()
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

/// What answers on the port the UI would use.
#[derive(Debug, PartialEq, Eq)]
enum Occupant {
    /// A healthy Izwi server of this version.
    Compatible,
    /// An Izwi server of another version, or one that is unwell. A second
    /// server next to it would share its data, so it is not started.
    OtherIzwi { version: String },
    /// Another program.
    Other(String),
}

/// The URL of the server the UI talks to, managed as app state once known.
pub struct ServerEndpoint(Url);

impl ServerEndpoint {
    pub fn new(url: Url) -> Self {
        Self(url)
    }

    pub fn url(&self) -> &Url {
        &self.0
    }
}

/// The server the UI talks to.
pub struct LocalServer {
    /// Differs from the requested URL when its port was taken.
    pub url: Url,
    /// Set when the desktop app started the server itself.
    pub managed: Option<ManagedServer>,
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

/// Use the Izwi server at `server_url`, starting one when the URL is local
/// and nothing answers there. When another program holds the port and
/// `may_change_port`, the server starts on a free port instead. An error's
/// outermost message is written for the user.
pub fn start_local_server<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    server_url: &Url,
    may_change_port: bool,
) -> Result<LocalServer> {
    let (host, mut port) = server_host_port(server_url)?;
    if !is_local_server_host(&host) {
        return Ok(LocalServer {
            url: server_url.clone(),
            managed: None,
        });
    }
    let bind_host = if host == "localhost" {
        "127.0.0.1".to_string()
    } else {
        host.clone()
    };
    let mut url = server_url.clone();

    let occupant = if is_server_reachable(&host, port, CONNECT_TIMEOUT) {
        Some(identify_occupant(server_url, IDENTIFY_TIMEOUT))
    } else if !port_is_free(&bind_host, port) {
        Some(Occupant::Other("the port is in use".to_string()))
    } else {
        None
    };
    // The previous version's server, on its way out after an update.
    let occupant = match occupant {
        Some(Occupant::OtherIzwi { version })
            if wait_for_port_to_close(&host, port, OTHER_IZWI_GRACE) =>
        {
            desktop_log::info(format!(
                "izwi-server {version} on port {port} has stopped; starting this version's server"
            ));
            None
        }
        // A server an older version left running (it had no owner pipe):
        // when it runs this app's own server binary, stop it.
        Some(Occupant::OtherIzwi { version })
            if stop_stale_server(&host, port, resolve_server_binary(app).as_deref()) =>
        {
            desktop_log::info(format!(
                "stopped izwi-server {version} left running on port {port}; starting this version's server"
            ));
            None
        }
        occupant => occupant,
    };
    match occupant {
        None => {}
        Some(Occupant::Compatible) => {
            desktop_log::warn(format!(
                "using compatible local izwi-server at {server_url}; it was not started by the desktop app and will not be stopped on exit"
            ));
            return Ok(LocalServer { url, managed: None });
        }
        Some(Occupant::OtherIzwi { version }) => {
            return Err(if version == env!("CARGO_PKG_VERSION") {
                anyhow!("Another Izwi server on port {port} isn't responding. Quit it, then open Izwi again.")
            } else {
                anyhow!("Izwi {version} is already running on port {port}. Quit it, then open Izwi again.")
            });
        }
        Some(Occupant::Other(reason)) if !may_change_port => {
            return Err(anyhow!("{reason}").context(format!(
                "Another program is using {server_url}. Stop it, or pass --server-url for the intended Izwi server."
            )));
        }
        Some(Occupant::Other(reason)) => {
            let free = first_free_port(&bind_host, FALLBACK_PORTS).with_context(|| {
                format!(
                    "Another program is using port {port}, and ports {}-{} are taken too.",
                    FALLBACK_PORTS.start(),
                    FALLBACK_PORTS.end()
                )
            })?;
            desktop_log::warn(format!(
                "port {port} is used by another program ({reason}); starting izwi-server on port {free}"
            ));
            port = free;
            url.set_port(Some(port))
                .map_err(|()| anyhow!("{url} cannot carry a port"))?;
        }
    }

    let spec = ServerSpawnSpec {
        binary: resolve_server_binary(app),
        bind_host,
        port,
    };
    let managed = spawn_server_process(app, &url, &spec)?;
    Ok(LocalServer {
        url,
        managed: Some(managed),
    })
}

/// Spawn the server child and wait until it answers the liveness probe.
fn spawn_server_process<R: tauri::Runtime>(
    app: &tauri::AppHandle<R>,
    server_url: &Url,
    spec: &ServerSpawnSpec,
) -> Result<ManagedServer> {
    const POLL_INTERVAL: Duration = Duration::from_millis(200);

    let mut cmd = match &spec.binary {
        Some(path) => Command::new(path),
        None => Command::new(platform_binary_name("izwi-server")),
    };

    let log_path = open_server_log(app, &mut cmd).context("Izwi couldn't start its server.")?;
    configure_local_server_command(&mut cmd, &spec.bind_host, spec.port);

    let mut child = cmd
        .spawn()
        .with_context(|| {
            format!(
                "failed to run {} for {}:{} (log: {})",
                cmd.get_program().to_string_lossy(),
                spec.bind_host,
                spec.port,
                log_path.display()
            )
        })
        .context("Izwi couldn't start its server.")?;

    let started = Instant::now();
    while started.elapsed() < START_TIMEOUT {
        if is_server_reachable(&spec.bind_host, spec.port, CONNECT_TIMEOUT)
            && identify_occupant(server_url, CONNECT_TIMEOUT) == Occupant::Compatible
        {
            desktop_log::info(format!(
                "izwi-server is ready on {}:{} after {:.1}s",
                spec.bind_host,
                spec.port,
                started.elapsed().as_secs_f64()
            ));
            return Ok(ManagedServer::new(child, log_path, spec.clone()));
        }

        if let Some(status) = child
            .try_wait()
            .context("failed while checking izwi-server status")?
        {
            return Err(anyhow!(
                "izwi-server exited before becoming ready on {}:{} ({}; log: {})",
                spec.bind_host,
                spec.port,
                status,
                log_path.display()
            ))
            .context("Izwi's server stopped while starting.");
        }

        thread::sleep(POLL_INTERVAL);
    }

    shutdown_child(&mut child);
    Err(anyhow!(
        "timed out waiting for izwi-server on {}:{} (log: {})",
        spec.bind_host,
        spec.port,
        log_path.display()
    ))
    .context(format!(
        "Izwi's server didn't start within {} seconds.",
        START_TIMEOUT.as_secs()
    ))
}

/// Keep the desktop-owned server running. A server that stops unexpectedly
/// is restarted after a pause; when it keeps stopping the monitor gives up
/// and tells the user. A server whose runtime poison persists (typically
/// because pinned residents keep the device from draining) is recreated: the
/// backstop for the OOM-recovery ladder, rate-limited with exponential
/// backoff so a server that poisons right after each restart cannot spin.
/// Exits when the shutdown flag is set at quit.
pub fn spawn_server_monitor(
    app: tauri::AppHandle,
    server_url: Url,
    managed: Arc<Mutex<Option<ManagedServer>>>,
    shutdown: Arc<AtomicBool>,
) {
    let _ = thread::Builder::new()
        .name("izwi-server-monitor".to_string())
        .spawn(move || {
            let mut restarts = RestartPolicy::default();
            let mut recreations = RecreationPolicy::default();
            let mut next_poison_poll = Instant::now() + POISON_POLL_INTERVAL;
            loop {
                thread::sleep(MONITOR_TICK);
                let Some(check) = check_server(&managed, &shutdown) else {
                    return;
                };
                if let Some(status) = check.exit_status {
                    desktop_log::warn(format!(
                        "izwi-server stopped unexpectedly ({status}); log: {}",
                        check.log_path.display()
                    ));
                    let Some(delay) = restarts.next_delay(Instant::now()) else {
                        give_up(&app);
                        return;
                    };
                    let restart = Restart {
                        app: &app,
                        server_url: &server_url,
                        spec: &check.spec,
                        managed: &managed,
                        shutdown: &shutdown,
                    };
                    if !restart.run(&mut restarts, delay) {
                        return;
                    }
                    continue;
                }

                if Instant::now() < next_poison_poll {
                    continue;
                }
                next_poison_poll = Instant::now() + POISON_POLL_INTERVAL;
                if poll_livez_runtime_poisoned(&server_url) != Some(true) {
                    // A healthy server resets the backoff ladder.
                    recreations.reset();
                    continue;
                }
                if !recreations.allows(Instant::now()) {
                    continue;
                }
                desktop_log::warn(
                    "izwi-server runtime is poisoned; recreating the desktop-owned server",
                );
                // Stop the poisoned child before respawning so the port frees.
                stop_current_server(&managed);
                let restart = Restart {
                    app: &app,
                    server_url: &server_url,
                    spec: &check.spec,
                    managed: &managed,
                    shutdown: &shutdown,
                };
                if !restart.run(&mut restarts, Duration::ZERO) {
                    return;
                }
            }
        });
}

struct ServerCheck {
    exit_status: Option<ExitStatus>,
    spec: ServerSpawnSpec,
    log_path: PathBuf,
}

/// The managed server's state, or None when the monitor should stop. Checked
/// under the lock: quitting sets the flag before it stops the server, so that
/// exit is never taken for a crash.
fn check_server(
    managed: &Mutex<Option<ManagedServer>>,
    shutdown: &AtomicBool,
) -> Option<ServerCheck> {
    let mut slot = managed.lock().ok()?;
    if shutdown.load(Ordering::Acquire) {
        return None;
    }
    let server = slot.as_mut()?;
    Some(ServerCheck {
        exit_status: server.exit_status(),
        spec: server.spec.clone(),
        log_path: server.log_path.clone(),
    })
}

fn stop_current_server(managed: &Mutex<Option<ManagedServer>>) {
    let child = managed
        .lock()
        .ok()
        .and_then(|mut slot| slot.as_mut().and_then(|server| server.child.take()));
    // Outside the lock: this can wait several seconds.
    if let Some(mut child) = child {
        shutdown_child(&mut child);
    }
}

struct Restart<'a> {
    app: &'a tauri::AppHandle,
    server_url: &'a Url,
    spec: &'a ServerSpawnSpec,
    managed: &'a Mutex<Option<ManagedServer>>,
    shutdown: &'a AtomicBool,
}

impl Restart<'_> {
    /// Start a replacement server after `delay`, retrying on the policy's
    /// pauses. False when the monitor should stop: the app is quitting, or
    /// restarting gave up.
    fn run(&self, policy: &mut RestartPolicy, mut delay: Duration) -> bool {
        loop {
            thread::sleep(delay);
            if self.shutdown.load(Ordering::Acquire) {
                return false;
            }
            match spawn_server_process(self.app, self.server_url, self.spec) {
                Ok(server) => {
                    desktop_log::info(format!(
                        "restarted izwi-server; log: {}",
                        server.log_path().display()
                    ));
                    return self.store(server);
                }
                Err(error) => {
                    desktop_log::warn(format!("could not restart izwi-server: {error:#}"));
                }
            }
            match policy.next_delay(Instant::now()) {
                Some(next) => delay = next,
                None => {
                    give_up(self.app);
                    return false;
                }
            }
        }
    }

    fn store(&self, mut server: ManagedServer) -> bool {
        let Ok(mut slot) = self.managed.lock() else {
            server.shutdown();
            return false;
        };
        if self.shutdown.load(Ordering::Acquire) {
            drop(slot);
            server.shutdown();
            return false;
        }
        *slot = Some(server);
        true
    }
}

fn give_up(app: &tauri::AppHandle) {
    desktop_log::error("izwi-server keeps stopping; not restarting it again");
    alerts::server_stopped(app);
}

/// Restarts allowed after unexpected stops: one per entry of
/// [`RESTART_DELAYS`] within [`RESTART_WINDOW`].
#[derive(Debug, Default)]
struct RestartPolicy {
    recent: Vec<Instant>,
}

impl RestartPolicy {
    /// The pause before the next restart, or None to give up.
    fn next_delay(&mut self, now: Instant) -> Option<Duration> {
        self.recent
            .retain(|at| now.saturating_duration_since(*at) < RESTART_WINDOW);
        let delay = *RESTART_DELAYS.get(self.recent.len())?;
        self.recent.push(now);
        Some(delay)
    }
}

/// Backoff between recreations of a poisoned server.
#[derive(Debug)]
struct RecreationPolicy {
    cooldown: Duration,
    last: Option<Instant>,
}

impl Default for RecreationPolicy {
    fn default() -> Self {
        Self {
            cooldown: RECREATION_MIN_COOLDOWN,
            last: None,
        }
    }
}

impl RecreationPolicy {
    fn reset(&mut self) {
        self.cooldown = RECREATION_MIN_COOLDOWN;
    }

    /// Whether to recreate now; each recreation doubles the next cooldown.
    fn allows(&mut self, now: Instant) -> bool {
        if self
            .last
            .is_some_and(|at| now.saturating_duration_since(at) < self.cooldown)
        {
            return false;
        }
        self.last = Some(now);
        self.cooldown = (self.cooldown * 2).min(RECREATION_MAX_COOLDOWN);
        true
    }
}

fn poll_livez_runtime_poisoned(server_url: &Url) -> Option<bool> {
    let live_url = server_url.join("/livez").ok()?;
    let body = reqwest::blocking::Client::builder()
        .timeout(POISON_PROBE_TIMEOUT)
        .build()
        .ok()?
        .get(live_url)
        .send()
        .ok()?
        // A permanently failed runtime answers 503 with the same body, so the
        // body is read whatever the status.
        .text()
        .ok()?;
    parse_livez_runtime_poisoned(&body)
}

#[derive(Debug, Deserialize)]
struct MonitorLiveResponse {
    #[serde(default)]
    runtime_poisoned: Option<bool>,
}

fn parse_livez_runtime_poisoned(body: &str) -> Option<bool> {
    let parsed: MonitorLiveResponse = serde_json::from_str(body).ok()?;
    parsed.runtime_poisoned
}

fn configure_local_server_command(cmd: &mut Command, bind_host: &str, port: u16) {
    // The standalone server reads the shared user config.toml and overlays its
    // inherited environment once. Do not synthesize default performance env
    // here: that would erase persisted opt-outs before the server can merge them.
    super::console::hide_console_window(cmd)
        .env("IZWI_HOST", bind_host)
        .env("IZWI_PORT", port.to_string())
        .env(DESKTOP_OWNER_PIPE_ENV, "1")
        .stdin(Stdio::piped());
    if std::env::var_os(SHUTDOWN_GRACE_ENV).is_none() {
        cmd.env(SHUTDOWN_GRACE_ENV, DESKTOP_SHUTDOWN_GRACE_SECS);
    }
    // macOS: libmalloc's large cache off, so a model load's freed host
    // buffers leave the server's footprint (izwi_env::allocator).
    izwi_env::apply_serving_allocator_env(cmd);
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

fn identify_occupant(server_url: &Url, timeout: Duration) -> Occupant {
    match fetch_liveness(server_url, timeout) {
        Ok((success, body)) => classify_liveness(success, &body, env!("CARGO_PKG_VERSION")),
        Err(error) => Occupant::Other(format!("{error:#}")),
    }
}

/// The `/livez` status and body. Runs during Tauri setup on the main thread,
/// inside the Tokio runtime, hence the separate thread.
fn fetch_liveness(server_url: &Url, timeout: Duration) -> Result<(bool, String)> {
    let live_url = server_url
        .join("/livez")
        .context("failed to construct izwi-server liveness URL")?;
    super::blocking::run_off_runtime(move || -> Result<(bool, String)> {
        let response = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .build()
            .context("failed to build izwi-server probe client")?
            .get(live_url)
            .send()
            .context("liveness probe failed")?;
        let success = response.status().is_success();
        // An unwell Izwi server answers with an error status and the same body.
        let body = response
            .text()
            .context("failed to read liveness response")?;
        Ok((success, body))
    })
}

fn classify_liveness(success: bool, body: &str, expected_version: &str) -> Occupant {
    let Ok(response) = serde_json::from_str::<LiveResponse>(body) else {
        return Occupant::Other("its liveness response was not Izwi's".to_string());
    };
    if success && response.status == "alive" && response.version == expected_version {
        Occupant::Compatible
    } else {
        Occupant::OtherIzwi {
            version: response.version,
        }
    }
}

fn wait_for_port_to_close(host: &str, port: u16, timeout: Duration) -> bool {
    let started = Instant::now();
    while is_server_reachable(host, port, CONNECT_TIMEOUT) {
        if started.elapsed() >= timeout {
            return false;
        }
        thread::sleep(Duration::from_millis(250));
    }
    true
}

fn port_is_free(bind_host: &str, port: u16) -> bool {
    TcpListener::bind((bind_host, port)).is_ok()
}

fn first_free_port(bind_host: &str, ports: impl IntoIterator<Item = u16>) -> Option<u16> {
    ports
        .into_iter()
        .find(|port| port_is_free(bind_host, *port))
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
    if child.try_wait().ok().flatten().is_some() {
        return;
    }
    request_graceful_termination(child);
    wait_or_kill(child);
}

fn wait_or_kill(child: &mut Child) {
    const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(8);
    const SHUTDOWN_POLL: Duration = Duration::from_millis(100);

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

/// Stop the server on `port` when it runs `ours`, and say whether the port
/// is free now. Any other program is left alone.
fn stop_stale_server(host: &str, port: u16, ours: Option<&Path>) -> bool {
    let Some(ours) = ours else {
        return false;
    };
    let Some(pid) = stale_server::listening_pid(port) else {
        return false;
    };
    match stale_server::process_executable(pid) {
        Some(running) if stale_server::is_our_server(&running, ours) => {}
        _ => return false,
    }
    stale_server::terminate(pid, false);
    if wait_for_port_to_close(host, port, STALE_SERVER_STOP) {
        return true;
    }
    stale_server::terminate(pid, true);
    wait_for_port_to_close(host, port, STALE_SERVER_STOP)
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

fn request_graceful_termination(child: &mut Child) {
    // Closing the owner pipe stops the server on every platform; Windows has
    // no SIGTERM, so without this it waited out the timeout and was killed.
    drop(child.stdin.take());

    // Only a running child: a reaped one's pid may belong to another process.
    #[cfg(unix)]
    if matches!(child.try_wait(), Ok(None)) {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(child.id().to_string())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
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
        assert_eq!(
            classify_liveness(
                true,
                r#"{"status":"alive","version":"0.1.0-test","uptime_secs":3}"#,
                "0.1.0-test",
            ),
            Occupant::Compatible
        );
    }

    #[test]
    fn liveness_probe_flags_other_izwi_versions_and_unwell_servers() {
        assert_eq!(
            classify_liveness(
                true,
                r#"{"status":"alive","version":"0.0.9","uptime_secs":3}"#,
                "0.1.0",
            ),
            Occupant::OtherIzwi {
                version: "0.0.9".to_string()
            }
        );
        assert_eq!(
            classify_liveness(
                false,
                r#"{"status":"alive","version":"0.1.0","runtime_poisoned":true}"#,
                "0.1.0",
            ),
            Occupant::OtherIzwi {
                version: "0.1.0".to_string()
            }
        );
    }

    #[test]
    fn liveness_probe_treats_non_izwi_responses_as_another_program() {
        for body in [r#"{"ok":true}"#, "<html>Jenkins</html>", ""] {
            assert!(matches!(
                classify_liveness(true, body, "0.1.0"),
                Occupant::Other(_)
            ));
        }
    }

    #[test]
    fn fallback_port_skips_ports_in_use() {
        let taken = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let taken_port = taken.local_addr().expect("addr").port();
        let free_port = TcpListener::bind(("127.0.0.1", 0))
            .and_then(|listener| listener.local_addr())
            .expect("free port")
            .port();

        assert!(!port_is_free("127.0.0.1", taken_port));
        assert_eq!(
            first_free_port("127.0.0.1", [taken_port, free_port]),
            Some(free_port)
        );
        assert_eq!(first_free_port("127.0.0.1", [taken_port]), None);
    }

    #[test]
    fn restarts_back_off_then_give_up_within_the_window() {
        let mut policy = RestartPolicy::default();
        let start = Instant::now();
        assert_eq!(policy.next_delay(start), Some(RESTART_DELAYS[0]));
        assert_eq!(policy.next_delay(start), Some(RESTART_DELAYS[1]));
        assert_eq!(policy.next_delay(start), Some(RESTART_DELAYS[2]));
        assert_eq!(policy.next_delay(start), None);
        // Stops spread out over more than the window are each restarted.
        assert_eq!(
            policy.next_delay(start + RESTART_WINDOW + Duration::from_secs(1)),
            Some(RESTART_DELAYS[0])
        );
    }

    #[test]
    fn poisoned_server_recreation_backs_off() {
        let mut policy = RecreationPolicy::default();
        let start = Instant::now();
        assert!(policy.allows(start));
        assert!(!policy.allows(start + RECREATION_MIN_COOLDOWN));
        assert!(policy.allows(start + RECREATION_MIN_COOLDOWN * 2));
        policy.reset();
        assert!(policy.allows(start + RECREATION_MIN_COOLDOWN * 3));
    }

    #[test]
    fn stopping_closes_the_owner_pipe() {
        let mut child = Command::new(if cfg!(windows) { "cmd" } else { "cat" })
            .args(if cfg!(windows) {
                &["/C", "more"][..]
            } else {
                &[][..]
            })
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .expect("spawn a process that reads stdin");
        request_graceful_termination(&mut child);
        assert!(child.stdin.is_none());
        let started = Instant::now();
        wait_or_kill(&mut child);
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn livez_poison_parsing_treats_missing_and_false_as_healthy() {
        assert_eq!(parse_livez_runtime_poisoned(r#"{"status":"alive","version":"0.1.0"}"#), None);
        assert_eq!(
            parse_livez_runtime_poisoned(
                r#"{"status":"alive","version":"0.1.0","runtime_poisoned":false}"#
            ),
            Some(false)
        );
        assert_eq!(
            parse_livez_runtime_poisoned(
                r#"{"status":"alive","version":"0.1.0","runtime_poisoned":true}"#
            ),
            Some(true)
        );
        assert_eq!(parse_livez_runtime_poisoned("not json"), None);
    }

    #[test]
    fn performance_startup_preserves_inherited_policy_and_leaves_file_defaults_to_server() {
        let mut command = Command::new("unused-server");
        command.env("IZWI_CUDA_MODE", "off");
        command.env("IZWI_CUDA_MTP_ADAPTIVE", "false");
        configure_local_server_command(&mut command, "127.0.0.1", 8080);
        let environment: std::collections::BTreeMap<_, _> = command.get_envs().collect();
        assert_eq!(
            environment[std::ffi::OsStr::new("IZWI_CUDA_MODE")],
            Some(std::ffi::OsStr::new("off"))
        );
        assert_eq!(
            environment[std::ffi::OsStr::new("IZWI_CUDA_MTP_ADAPTIVE")],
            Some(std::ffi::OsStr::new("false"))
        );
        assert!(!environment.contains_key(std::ffi::OsStr::new("IZWI_LOADING_MODE")));
        assert!(command.get_args().next().is_none());
    }

    #[test]
    fn desktop_server_drains_briefly_at_quit() {
        let mut command = Command::new("unused-server");
        configure_local_server_command(&mut command, "127.0.0.1", 8080);
        let environment: std::collections::BTreeMap<_, _> = command.get_envs().collect();
        if std::env::var_os(SHUTDOWN_GRACE_ENV).is_none() {
            assert_eq!(
                environment[std::ffi::OsStr::new(SHUTDOWN_GRACE_ENV)],
                Some(std::ffi::OsStr::new(DESKTOP_SHUTDOWN_GRACE_SECS))
            );
        }
        assert_eq!(
            environment[std::ffi::OsStr::new(DESKTOP_OWNER_PIPE_ENV)],
            Some(std::ffi::OsStr::new("1"))
        );
    }

    #[test]
    fn waits_for_a_closing_port() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let port = listener.local_addr().expect("addr").port();
        assert!(!wait_for_port_to_close(
            "127.0.0.1",
            port,
            Duration::from_millis(300)
        ));
        drop(listener);
        assert!(wait_for_port_to_close(
            "127.0.0.1",
            port,
            Duration::from_secs(2)
        ));
    }

    #[test]
    fn local_server_starts_with_the_serving_allocator() {
        let mut command = Command::new("unused-server");
        configure_local_server_command(&mut command, "127.0.0.1", 8080);
        let environment: std::collections::BTreeMap<_, _> = command.get_envs().collect();
        for (name, value) in izwi_env::serving_allocator_env() {
            if std::env::var_os(name).is_none() {
                assert_eq!(
                    environment[std::ffi::OsStr::new(name)],
                    Some(std::ffi::OsStr::new(value))
                );
            }
        }
    }
}
