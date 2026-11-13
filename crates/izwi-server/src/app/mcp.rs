//! A Model Context Protocol client for agent turns: the tools of an MCP
//! server as tools a reply may call, over the Streamable HTTP transport or,
//! for a server that runs as a local command, over its standard streams.
//!
//! A session names its servers. One it names by URL is reached only when
//! its host is in `IZWI_MCP_ALLOWED_HOSTS`, so API clients cannot point the
//! server at arbitrary addresses on its network. One it names by label
//! alone is a server the operator set up in the MCP servers file
//! ([`ConfiguredServers`]): only there can a command be named, so no client
//! request ever chooses what runs. The client speaks the request/response
//! subset an agent needs: `initialize`, `tools/list` and `tools/call`. Over
//! HTTP a response may arrive as JSON or as a server-sent event stream, and
//! requests the MCP server sends are not answered; over standard streams a
//! `ping` is answered and any other request is refused.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use futures::StreamExt;
use izwi_agent::{AgentError, Tool, ToolContext, ToolOutput, ToolRegistry};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::sync::oneshot;

/// The protocol revision this client speaks.
const PROTOCOL_VERSION: &str = "2025-06-18";
pub(crate) const ALLOWED_HOSTS_ENV: &str = "IZWI_MCP_ALLOWED_HOSTS";
pub(crate) const SERVERS_FILE_ENV: &str = "IZWI_MCP_SERVERS_FILE";
/// The MCP servers file's name in the data directory.
const SERVERS_FILE_NAME: &str = "mcp.json";
/// How long a local server may take to start and answer `initialize`.
const STDIO_START_TIMEOUT: Duration = Duration::from_secs(60);
/// The most MCP servers one session names.
pub(crate) const MAX_SERVERS: usize = 8;
/// The most tools taken from one server.
const MAX_TOOLS_PER_SERVER: usize = 64;
/// The most of a response the client reads.
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
/// The most of a tool's result a reply reads.
const MAX_RESULT_CHARS: usize = 16_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// An MCP server a session names.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub(crate) struct McpServerConfig {
    /// Names the server in events and errors.
    #[serde(alias = "server_label")]
    pub(crate) label: String,
    /// Where the server is. Absent, the label names one of the servers in
    /// the operator's MCP servers file.
    #[serde(default, alias = "server_url")]
    pub(crate) url: Option<String>,
    /// The configured server to use when it is not the one the label
    /// names: OpenAI's `connector_id`, here a label in the MCP servers file.
    #[serde(default)]
    pub(crate) connector_id: Option<String>,
    /// Sent with every request, for example `Authorization`.
    #[serde(default)]
    pub(crate) headers: std::collections::BTreeMap<String, String>,
    /// The tools a reply may call; all of the server's when absent.
    #[serde(default)]
    pub(crate) allowed_tools: Option<Vec<String>>,
}

/// The hosts `IZWI_MCP_ALLOWED_HOSTS` names: `host` (any port), `host:port`,
/// or `*` for any host. Empty turns MCP off.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct AllowedHosts(Vec<String>);

impl AllowedHosts {
    pub(crate) fn from_env() -> Self {
        Self::parse(&std::env::var(ALLOWED_HOSTS_ENV).unwrap_or_default())
    }

    pub(crate) fn parse(list: &str) -> Self {
        Self(
            list.split(',')
                .map(|host| host.trim().to_ascii_lowercase())
                .filter(|host| !host.is_empty())
                .collect(),
        )
    }

    /// `url` parsed, when the server may connect to it.
    fn check(&self, label: &str, url: &str) -> Result<reqwest::Url, String> {
        if self.0.is_empty() {
            return Err(format!(
                "MCP servers are off: set {ALLOWED_HOSTS_ENV} to the hosts this server may connect to, or name a server from its MCP servers file"
            ));
        }
        let url = parse_http_url(label, url)?;
        let host = url.host_str().unwrap_or_default().to_ascii_lowercase();
        let with_port = url
            .port_or_known_default()
            .map(|port| format!("{host}:{port}"));
        let allowed = self
            .0
            .iter()
            .any(|entry| entry == "*" || *entry == host || Some(entry) == with_port.as_ref());
        if !allowed {
            return Err(format!(
                "MCP server `{label}` is on host `{host}`, which {ALLOWED_HOSTS_ENV} does not allow"
            ));
        }
        Ok(url)
    }
}

fn parse_http_url(label: &str, url: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(url)
        .map_err(|err| format!("MCP server `{label}` has an invalid URL: {err}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("MCP server `{label}` must use http or https"));
    }
    Ok(url)
}

fn check_label(label: &str) -> Result<(), String> {
    if label.is_empty()
        || !label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "MCP server label {label:?} must be letters, digits, `_` or `-`"
        ));
    }
    Ok(())
}

/// A server in the operator's MCP servers file: a local command that speaks
/// MCP over its standard streams, or a remote server.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub(crate) enum ConfiguredServer {
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        /// Added to the server's own environment.
        #[serde(default)]
        env: BTreeMap<String, String>,
        #[serde(default)]
        cwd: Option<PathBuf>,
    },
    Http {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
    },
}

impl ConfiguredServer {
    pub(crate) fn transport(&self) -> &'static str {
        match self {
            Self::Stdio { .. } => "stdio",
            Self::Http { .. } => "http",
        }
    }
}

/// The MCP servers the operator set up, in the file `IZWI_MCP_SERVERS_FILE`
/// names (by default `mcp.json` in the data directory), in the form other
/// MCP clients use:
///
/// ```json
/// {"mcpServers": {
///   "files": {"command": "/usr/local/bin/npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]},
///   "docs": {"url": "https://mcp.example.com/mcp", "headers": {"Authorization": "Bearer ..."}}
/// }}
/// ```
///
/// A session uses one by naming its label without a URL. Each is connected
/// when first used and shared by every session after: a local server is one
/// process, started again if it has exited.
#[derive(Default)]
pub(crate) struct ConfiguredServers {
    servers: BTreeMap<String, ConfiguredServer>,
    clients: tokio::sync::Mutex<HashMap<String, Arc<McpClient>>>,
}

impl std::fmt::Debug for ConfiguredServers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Commands, environments and headers may carry credentials.
        f.debug_list().entries(self.servers.keys()).finish()
    }
}

#[derive(Deserialize)]
struct ServersFile {
    #[serde(default, rename = "mcpServers")]
    mcp_servers: BTreeMap<String, ConfiguredServer>,
}

impl ConfiguredServers {
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let file: ServersFile = serde_json::from_str(text).map_err(|err| err.to_string())?;
        for (label, server) in &file.mcp_servers {
            check_label(label)?;
            match server {
                ConfiguredServer::Stdio { command, .. } if command.trim().is_empty() => {
                    return Err(format!("MCP server `{label}` has an empty command"));
                }
                ConfiguredServer::Http { url, .. } => {
                    parse_http_url(label, url)?;
                }
                ConfiguredServer::Stdio { .. } => {}
            }
        }
        Ok(Self {
            servers: file.mcp_servers,
            clients: Default::default(),
        })
    }

    /// The servers of the file at `path`: none when there is no file. A
    /// file that cannot be read as one is reported and sets up none.
    fn load(path: &Path) -> Self {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Self::default(),
            Err(err) => {
                tracing::warn!(path = %path.display(), "MCP servers file not read: {err}");
                return Self::default();
            }
        };
        match Self::parse(&text) {
            Ok(servers) => {
                tracing::info!(path = %path.display(), servers = servers.servers.len(), "MCP servers file loaded");
                servers
            }
            Err(err) => {
                tracing::warn!(path = %path.display(), "MCP servers file ignored: {err}");
                Self::default()
            }
        }
    }

    /// The process's configured servers, read once.
    pub(crate) fn shared() -> Arc<Self> {
        static SHARED: OnceLock<Arc<ConfiguredServers>> = OnceLock::new();
        SHARED
            .get_or_init(|| {
                let path = std::env::var_os(SERVERS_FILE_ENV)
                    .filter(|path| !path.is_empty())
                    .map(PathBuf::from)
                    .unwrap_or_else(|| {
                        izwi_db::storage_layout::resolve_db_path().with_file_name(SERVERS_FILE_NAME)
                    });
                Arc::new(Self::load(&path))
            })
            .clone()
    }

    /// Each server's label and transport (`stdio` or `http`).
    pub(crate) fn list(&self) -> Vec<(&str, &'static str)> {
        self.servers
            .iter()
            .map(|(label, server)| (label.as_str(), server.transport()))
            .collect()
    }

    /// The connection to the server labelled `label`. `headers` are a
    /// session's own (a user's token for a remote server): with them the
    /// connection is the session's, not the shared one.
    async fn client(
        &self,
        label: &str,
        headers: &BTreeMap<String, String>,
    ) -> Result<Arc<McpClient>, String> {
        let Some(server) = self.servers.get(label) else {
            return Err(format!(
                "MCP server `{label}` has no URL and is not in this server's MCP servers file ({SERVERS_FILE_ENV})"
            ));
        };
        if !headers.is_empty() {
            let ConfiguredServer::Http { url, headers: own } = server else {
                return Err(format!(
                    "MCP server `{label}` runs as a local command and takes no headers"
                ));
            };
            let mut merged = own.clone();
            merged.extend(headers.clone());
            return McpClient::connect_http(label, parse_http_url(label, url)?, &merged).await;
        }
        let mut clients = self.clients.lock().await;
        if let Some(client) = clients.get(label).filter(|client| !client.is_closed()) {
            return Ok(client.clone());
        }
        let client = match server {
            ConfiguredServer::Http { url, headers } => {
                McpClient::connect_http(label, parse_http_url(label, url)?, headers).await?
            }
            ConfiguredServer::Stdio {
                command,
                args,
                env,
                cwd,
            } => McpClient::spawn(label, command, args, env, cwd.as_deref()).await?,
        };
        clients.insert(label.to_string(), client.clone());
        Ok(client)
    }
}

/// What a session's MCP servers may be: hosts it may name by URL, and the
/// operator's configured servers it may name by label.
#[derive(Debug, Clone, Default)]
pub(crate) struct McpAccess {
    pub(crate) allowed: AllowedHosts,
    pub(crate) configured: Arc<ConfiguredServers>,
}

impl McpAccess {
    pub(crate) fn from_env() -> Self {
        Self {
            allowed: AllowedHosts::from_env(),
            configured: ConfiguredServers::shared(),
        }
    }
}

/// A tool an MCP server offers.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct McpToolInfo {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) input_schema: Value,
}

/// A connection to one MCP server.
pub(crate) struct McpClient {
    label: String,
    transport: Transport,
    next_id: AtomicU64,
}

enum Transport {
    Http(HttpTransport),
    Stdio(StdioTransport),
}

struct HttpTransport {
    url: reqwest::Url,
    http: reqwest::Client,
    headers: HeaderMap,
    session_id: Mutex<Option<String>>,
    protocol_version: Mutex<String>,
}

/// A local server process: one JSON-RPC message a line on its standard
/// input and output. The process is killed when the connection drops.
struct StdioTransport {
    stdin: StdioInput,
    /// Requests waiting for the response with their id.
    waiting: Waiting,
    closed: Arc<AtomicBool>,
    _child: tokio::process::Child,
    reader: tokio::task::JoinHandle<()>,
}

type Waiting = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;
type StdioInput = Arc<tokio::sync::Mutex<tokio::process::ChildStdin>>;

impl Drop for StdioTransport {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Headers, commands and environments may carry credentials.
        f.debug_struct("McpClient")
            .field("label", &self.label)
            .field(
                "transport",
                &match &self.transport {
                    Transport::Http(http) => http.url.as_str(),
                    Transport::Stdio(_) => "stdio",
                },
            )
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Connect to the server a session's `config` names and complete the
    /// protocol handshake.
    pub(crate) async fn connect(
        config: &McpServerConfig,
        access: &McpAccess,
    ) -> Result<Arc<Self>, String> {
        let label = config.label.trim();
        check_label(label)?;
        match &config.url {
            Some(url) => {
                let url = access.allowed.check(label, url)?;
                Self::connect_http(label, url, &config.headers).await
            }
            None => {
                let name = config.connector_id.as_deref().unwrap_or(label);
                access.configured.client(name, &config.headers).await
            }
        }
    }

    async fn connect_http(
        label: &str,
        url: reqwest::Url,
        given: &BTreeMap<String, String>,
    ) -> Result<Arc<Self>, String> {
        let mut headers = HeaderMap::new();
        for (name, value) in given {
            let name = HeaderName::try_from(name.as_str())
                .map_err(|_| format!("MCP server `{label}` has an invalid header name"))?;
            let mut value = HeaderValue::try_from(value.as_str())
                .map_err(|_| format!("MCP server `{label}` has an invalid header value"))?;
            value.set_sensitive(true);
            headers.insert(name, value);
        }
        // reqwest is built without a TLS provider of its own.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let http = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            // A redirect could leave the allowed hosts.
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| format!("MCP client could not start: {err}"))?;
        let client = Arc::new(Self {
            label: label.to_string(),
            transport: Transport::Http(HttpTransport {
                url,
                http,
                headers,
                session_id: Mutex::new(None),
                protocol_version: Mutex::new(PROTOCOL_VERSION.to_string()),
            }),
            next_id: AtomicU64::new(1),
        });
        client.initialize().await?;
        Ok(client)
    }

    /// Start a local server and complete the protocol handshake.
    async fn spawn(
        label: &str,
        command: &str,
        args: &[String],
        env: &BTreeMap<String, String>,
        cwd: Option<&Path>,
    ) -> Result<Arc<Self>, String> {
        let mut process = tokio::process::Command::new(command);
        process
            .args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        if let Some(cwd) = cwd {
            process.current_dir(cwd);
        }
        let mut child = process
            .spawn()
            .map_err(|err| format!("MCP server `{label}` could not start `{command}`: {err}"))?;
        let (Some(stdin), Some(stdout)) = (child.stdin.take(), child.stdout.take()) else {
            return Err(format!("MCP server `{label}` has no standard streams"));
        };
        let waiting = Waiting::default();
        let closed = Arc::new(AtomicBool::new(false));
        let stdin = Arc::new(tokio::sync::Mutex::new(stdin));
        let reader = tokio::spawn(read_stdio(
            label.to_string(),
            stdout,
            stdin.clone(),
            waiting.clone(),
            closed.clone(),
        ));
        let client = Arc::new(Self {
            label: label.to_string(),
            transport: Transport::Stdio(StdioTransport {
                stdin,
                waiting,
                closed,
                _child: child,
                reader,
            }),
            next_id: AtomicU64::new(1),
        });
        client.initialize().await?;
        Ok(client)
    }

    async fn initialize(&self) -> Result<(), String> {
        let initialized = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "izwi", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;
        if let (Transport::Http(http), Some(version)) =
            (&self.transport, initialized["protocolVersion"].as_str())
        {
            *lock(&http.protocol_version) = version.to_string();
        }
        self.notify("notifications/initialized").await
    }

    /// Whether the connection can no longer answer: a local server exited.
    fn is_closed(&self) -> bool {
        match &self.transport {
            Transport::Http(_) => false,
            Transport::Stdio(stdio) => stdio.closed.load(Ordering::Relaxed),
        }
    }

    /// The server's tools, narrowed to `allowed` when given.
    pub(crate) async fn list_tools(
        &self,
        allowed: Option<&[String]>,
    ) -> Result<Vec<McpToolInfo>, String> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(cursor) => json!({"cursor": cursor}),
                None => json!({}),
            };
            let page = self.request("tools/list", params).await?;
            for tool in page["tools"].as_array().into_iter().flatten() {
                let Some(name) = tool["name"].as_str() else {
                    continue;
                };
                if allowed.is_some_and(|allowed| !allowed.iter().any(|a| a == name)) {
                    continue;
                }
                if tools.len() == MAX_TOOLS_PER_SERVER {
                    return Err(format!(
                        "MCP server `{}` offers more than {MAX_TOOLS_PER_SERVER} tools; name the ones to use in `allowed_tools`",
                        self.label
                    ));
                }
                tools.push(McpToolInfo {
                    name: name.to_string(),
                    description: tool["description"].as_str().unwrap_or_default().to_string(),
                    input_schema: match &tool["inputSchema"] {
                        schema @ Value::Object(_) => schema.clone(),
                        _ => json!({"type": "object", "properties": {}}),
                    },
                });
            }
            cursor = page["nextCursor"]
                .as_str()
                .filter(|cursor| !cursor.is_empty())
                .map(str::to_string);
            if cursor.is_none() {
                return Ok(tools);
            }
        }
    }

    /// Call `name`: `Ok` with the result's text, `Err` with the tool's or
    /// the protocol's error.
    pub(crate) async fn call_tool(&self, name: &str, arguments: Value) -> Result<String, String> {
        let result = self
            .request("tools/call", json!({"name": name, "arguments": arguments}))
            .await?;
        let text = result_text(&result);
        if result["isError"].as_bool() == Some(true) {
            return Err(text);
        }
        Ok(text)
    }

    async fn request(&self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let failed = |what: String| format!("MCP server `{}` {method}: {what}", self.label);
        let message = match &self.transport {
            Transport::Http(http) => http.request(&body, id, method != "initialize").await,
            // A first run may install the server before it can answer.
            Transport::Stdio(stdio) if method == "initialize" => {
                stdio.request(&body, id, STDIO_START_TIMEOUT).await
            }
            Transport::Stdio(stdio) => stdio.request(&body, id, REQUEST_TIMEOUT).await,
        }
        .map_err(failed)?;
        if let Some(error) = message.get("error") {
            return Err(failed(
                error["message"]
                    .as_str()
                    .unwrap_or("the server reported an error")
                    .to_string(),
            ));
        }
        Ok(message["result"].clone())
    }

    async fn notify(&self, method: &str) -> Result<(), String> {
        let body = json!({"jsonrpc": "2.0", "method": method});
        match &self.transport {
            Transport::Http(http) => match http.post(&body, true).await {
                Ok(response) if response.status().is_success() => Ok(()),
                Ok(response) => Err(format!("HTTP {}", response.status().as_u16())),
                Err(err) => Err(err.to_string()),
            },
            Transport::Stdio(stdio) => stdio.send(&body).await,
        }
        .map_err(|err| format!("MCP server `{}` {method}: {err}", self.label))
    }
}

impl HttpTransport {
    /// Send a request and read the response to `id`.
    async fn request(&self, body: &Value, id: u64, negotiated: bool) -> Result<Value, String> {
        let response = self
            .post(body, negotiated)
            .await
            .map_err(|err| err.to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status().as_u16()));
        }
        if let Some(session) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
        {
            *lock(&self.session_id) = Some(session.to_string());
        }
        let event_stream = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|kind| kind.starts_with("text/event-stream"));
        if event_stream {
            read_event_stream(response, id).await
        } else {
            let bytes = read_capped(response).await?;
            serde_json::from_slice::<Value>(&bytes).map_err(|err| format!("invalid JSON: {err}"))
        }
    }

    async fn post(&self, body: &Value, negotiated: bool) -> reqwest::Result<reqwest::Response> {
        let mut request = self
            .http
            .post(self.url.clone())
            .headers(self.headers.clone())
            .header(ACCEPT, "application/json, text/event-stream")
            .json(body);
        if negotiated {
            request = request.header("mcp-protocol-version", lock(&self.protocol_version).clone());
        }
        if let Some(session) = lock(&self.session_id).clone() {
            request = request.header("mcp-session-id", session);
        }
        request.send().await
    }
}

impl StdioTransport {
    /// Send a request and wait for the response to `id`.
    async fn request(&self, body: &Value, id: u64, timeout: Duration) -> Result<Value, String> {
        let (answer, answered) = oneshot::channel();
        lock(&self.waiting).insert(id, answer);
        let sent = self.send(body).await;
        let result = match sent {
            Err(err) => Err(err),
            Ok(()) => match tokio::time::timeout(timeout, answered).await {
                Ok(Ok(message)) => Ok(message),
                Ok(Err(_)) => Err("the server process exited".to_string()),
                Err(_) => Err(format!("no answer within {} seconds", timeout.as_secs())),
            },
        };
        lock(&self.waiting).remove(&id);
        result
    }

    async fn send(&self, message: &Value) -> Result<(), String> {
        if self.closed.load(Ordering::Relaxed) {
            return Err("the server process exited".to_string());
        }
        write_line(&self.stdin, message).await
    }
}

async fn write_line(stdin: &StdioInput, message: &Value) -> Result<(), String> {
    let mut line = message.to_string();
    line.push('\n');
    let mut stdin = stdin.lock().await;
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|err| err.to_string())?;
    stdin.flush().await.map_err(|err| err.to_string())
}

/// Read a local server's output a line at a time: a response goes to the
/// request waiting for its id, a `ping` is answered, any other request is
/// refused, and notifications are dropped. Ends when the server closes its
/// output, which fails every request still waiting.
async fn read_stdio(
    label: String,
    stdout: tokio::process::ChildStdout,
    stdin: StdioInput,
    waiting: Waiting,
    closed: Arc<AtomicBool>,
) {
    let mut lines = tokio::io::BufReader::new(stdout);
    let mut line = Vec::new();
    loop {
        line.clear();
        let mut limited = (&mut lines).take(MAX_RESPONSE_BYTES as u64 + 1);
        match limited.read_until(b'\n', &mut line).await {
            Ok(0) | Err(_) => break,
            Ok(_) if line.len() > MAX_RESPONSE_BYTES => {
                tracing::warn!(server = %label, "MCP server wrote a line over {MAX_RESPONSE_BYTES} bytes; closing it");
                break;
            }
            Ok(_) => {}
        }
        let Ok(message) = serde_json::from_slice::<Value>(&line) else {
            // Not protocol output: a server that logs to stdout.
            continue;
        };
        let is_response = message.get("result").is_some() || message.get("error").is_some();
        if is_response {
            if let Some(answer) = message["id"]
                .as_u64()
                .and_then(|id| lock(&waiting).remove(&id))
            {
                let _ = answer.send(message);
            }
        } else if let (Some(method), Some(id)) = (message["method"].as_str(), message.get("id")) {
            let reply = if method == "ping" {
                json!({"jsonrpc": "2.0", "id": id, "result": {}})
            } else {
                json!({"jsonrpc": "2.0", "id": id,
                       "error": {"code": -32601, "message": "Method not found"}})
            };
            if write_line(&stdin, &reply).await.is_err() {
                break;
            }
        }
    }
    closed.store(true, Ordering::Relaxed);
    // Dropping the senders fails the requests still waiting.
    lock(&waiting).clear();
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poison| poison.into_inner())
}

async fn read_capped(response: reqwest::Response) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| err.to_string())?;
        if body.len() + chunk.len() > MAX_RESPONSE_BYTES {
            return Err(format!("the response is over {MAX_RESPONSE_BYTES} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Read a server-sent event stream until the JSON-RPC response to `id`.
/// Notifications and the server's own requests on the stream are skipped.
async fn read_event_stream(response: reqwest::Response, id: u64) -> Result<Value, String> {
    let mut events = EventStream::default();
    let mut read = 0usize;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| err.to_string())?;
        read += chunk.len();
        if read > MAX_RESPONSE_BYTES {
            return Err(format!("the response is over {MAX_RESPONSE_BYTES} bytes"));
        }
        for data in events.push(&chunk) {
            if let Some(message) = response_to(&data, id) {
                return Ok(message);
            }
        }
    }
    events
        .finish()
        .and_then(|data| response_to(&data, id))
        .ok_or_else(|| "the stream ended without a response".to_string())
}

fn response_to(data: &str, id: u64) -> Option<Value> {
    let message = serde_json::from_str::<Value>(data).ok()?;
    let is_response = message.get("result").is_some() || message.get("error").is_some();
    (is_response && message["id"].as_u64() == Some(id)).then_some(message)
}

/// Splits a server-sent event stream into the `data` of its events.
#[derive(Default)]
struct EventStream {
    buffer: Vec<u8>,
    data: Vec<String>,
}

impl EventStream {
    /// The data of the events `chunk` completes.
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(end) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=end).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if let Some(data) = self.finish() {
                    events.push(data);
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data
                    .push(data.strip_prefix(' ').unwrap_or(data).to_string());
            }
        }
        events
    }

    /// The data of an event the stream ended in the middle of.
    fn finish(&mut self) -> Option<String> {
        (!self.data.is_empty()).then(|| std::mem::take(&mut self.data).join("\n"))
    }
}

/// A tool result as text a reply can read: its text parts, a note for each
/// part that is not text, or its structured content when it has no parts.
fn result_text(result: &Value) -> String {
    let mut parts = Vec::new();
    for part in result["content"].as_array().into_iter().flatten() {
        match part["type"].as_str() {
            Some("text") => parts.push(part["text"].as_str().unwrap_or_default().to_string()),
            Some(kind) => parts.push(format!("[{kind} content]")),
            None => {}
        }
    }
    let mut text = parts.join("\n");
    if text.trim().is_empty() {
        if let Some(structured) = result.get("structuredContent").filter(|v| !v.is_null()) {
            text = structured.to_string();
        }
    }
    if text.chars().count() > MAX_RESULT_CHARS {
        text = text.chars().take(MAX_RESULT_CHARS).collect::<String>() + "\n[result cut short]";
    }
    text
}

/// One tool of an MCP server as a tool a turn runs.
struct McpTool {
    client: Arc<McpClient>,
    info: McpToolInfo,
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.info.name
    }

    fn description(&self) -> &str {
        &self.info.description
    }

    fn parameters(&self) -> Value {
        self.info.input_schema.clone()
    }

    fn executor(&self) -> &'static str {
        "mcp"
    }

    async fn invoke(&self, ctx: ToolContext) -> izwi_agent::Result<ToolOutput> {
        self.client
            .call_tool(&self.info.name, ctx.arguments)
            .await
            .map(|text| ToolOutput { text })
            .map_err(AgentError::Tool)
    }
}

/// A session's MCP tools: connected once, added to each turn's registry.
#[derive(Clone, Default)]
pub(crate) struct McpTools {
    tools: Vec<McpToolEntry>,
}

#[derive(Clone)]
struct McpToolEntry {
    /// The label the session gave the tool's server.
    server: String,
    client: Arc<McpClient>,
    info: McpToolInfo,
}

impl PartialEq for McpTools {
    fn eq(&self, other: &Self) -> bool {
        self.tools.len() == other.tools.len()
            && self.tools.iter().zip(&other.tools).all(|(a, b)| {
                a.server == b.server && Arc::ptr_eq(&a.client, &b.client) && a.info == b.info
            })
    }
}

impl std::fmt::Debug for McpTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

impl McpTools {
    /// Connect to each of `servers` and list its tools. Two servers may not
    /// offer a tool of the same name.
    pub(crate) async fn connect(
        servers: &[McpServerConfig],
        access: &McpAccess,
    ) -> Result<Self, String> {
        if servers.len() > MAX_SERVERS {
            return Err(format!(
                "A session names at most {MAX_SERVERS} MCP servers, not {}",
                servers.len()
            ));
        }
        let mut tools: Vec<McpToolEntry> = Vec::new();
        for config in servers {
            let client = McpClient::connect(config, access).await?;
            for info in client.list_tools(config.allowed_tools.as_deref()).await? {
                if tools.iter().any(|other| other.info.name == info.name) {
                    return Err(format!(
                        "MCP tool `{}` is offered twice; narrow one server with `allowed_tools`",
                        info.name
                    ));
                }
                tools.push(McpToolEntry {
                    server: config.label.trim().to_string(),
                    client: client.clone(),
                    info,
                });
            }
        }
        Ok(Self { tools })
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The tools as OpenAI function tools, for a turn that offers them but
    /// leaves running them to its caller.
    pub(crate) fn function_definitions(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|tool| {
                izwi_agent::function_definition(
                    &tool.info.name,
                    &tool.info.description,
                    tool.info.input_schema.clone(),
                )
            })
            .collect()
    }

    /// The label of the server that offers `name`.
    pub(crate) fn server_of(&self, name: &str) -> Option<&str> {
        self.tools
            .iter()
            .find(|tool| tool.info.name == name)
            .map(|tool| tool.server.as_str())
    }

    /// The tools of the server labelled `label`.
    pub(crate) fn tools_of(&self, label: &str) -> Vec<&McpToolInfo> {
        self.tools
            .iter()
            .filter(|tool| tool.server == label)
            .map(|tool| &tool.info)
            .collect()
    }

    /// Call `name` with `arguments` (a JSON object as text).
    pub(crate) async fn call(&self, name: &str, arguments: &str) -> Result<String, String> {
        let Some(tool) = self.tools.iter().find(|tool| tool.info.name == name) else {
            return Err(format!("No MCP tool named `{name}` is available."));
        };
        let arguments = match arguments.trim() {
            "" => json!({}),
            arguments => serde_json::from_str::<Value>(arguments)
                .map_err(|err| format!("Invalid arguments: {err}"))?,
        };
        tool.client.call_tool(name, arguments).await
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|tool| tool.info.name.as_str())
    }

    /// Add the tools to `registry`. `Err` names a tool the registry already
    /// has.
    pub(crate) fn register(&self, registry: &mut ToolRegistry) -> Result<(), String> {
        let taken = registry.tool_names();
        for tool in &self.tools {
            if taken.contains(&tool.info.name) {
                return Err(format!(
                    "MCP tool `{}` has the name of another tool of the session",
                    tool.info.name
                ));
            }
            registry.register(McpTool {
                client: tool.client.clone(),
                info: tool.info.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::Router;

    /// A small MCP server: two tools, answering as JSON or as an event
    /// stream, with a session id and a required bearer token.
    pub(crate) async fn serve(event_stream: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let handler = move |headers: axum::http::HeaderMap, body: String| {
            let log = log.clone();
            async move {
                if headers.get("authorization").and_then(|v| v.to_str().ok()) != Some("Bearer t") {
                    return StatusCode::UNAUTHORIZED.into_response();
                }
                let message: Value = serde_json::from_str(&body).unwrap();
                let method = message["method"].as_str().unwrap_or_default().to_string();
                lock(&log).push(json!({
                    "method": method,
                    "params": message["params"],
                    "session": headers.get("mcp-session-id").and_then(|v| v.to_str().ok()),
                    "version": headers.get("mcp-protocol-version").and_then(|v| v.to_str().ok()),
                }));
                let Some(id) = message.get("id").cloned() else {
                    return StatusCode::ACCEPTED.into_response();
                };
                let result = match method.as_str() {
                    "initialize" => json!({
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    }),
                    "tools/list" if message["params"]["cursor"].is_null() => json!({
                        "tools": [{
                            "name": "add",
                            "description": "Add two numbers.",
                            "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}}},
                        }],
                        "nextCursor": "page-2",
                    }),
                    "tools/list" => json!({"tools": [{"name": "fail"}]}),
                    "tools/call" if message["params"]["name"] == "add" => json!({
                        "content": [{"type": "text", "text": "5"}, {"type": "image", "data": ""}],
                    }),
                    "tools/call" => json!({
                        "content": [{"type": "text", "text": "it broke"}],
                        "isError": true,
                    }),
                    _ => Value::Null,
                };
                let reply = json!({"jsonrpc": "2.0", "id": id, "result": result});
                let mut response = if event_stream {
                    // A notification first, and the response split over
                    // two data lines' worth of bytes.
                    let body = format!(
                        "event: message\ndata: {}\n\nevent: message\ndata: {reply}\n\n",
                        json!({"jsonrpc": "2.0", "method": "notifications/progress"})
                    );
                    ([(CONTENT_TYPE, "text/event-stream")], body).into_response()
                } else {
                    axum::Json(reply).into_response()
                };
                if method == "initialize" {
                    response
                        .headers_mut()
                        .insert("mcp-session-id", HeaderValue::from_static("sess-1"));
                }
                response
            }
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/mcp", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(listener, Router::new().route("/mcp", post(handler)))
                .await
                .unwrap();
        });
        (url, seen)
    }

    pub(crate) fn hosts(list: &str) -> McpAccess {
        McpAccess {
            allowed: AllowedHosts::parse(list),
            configured: Default::default(),
        }
    }

    pub(crate) fn config(url: &str) -> McpServerConfig {
        McpServerConfig {
            label: "calc".to_string(),
            url: Some(url.to_string()),
            connector_id: None,
            headers: [("Authorization".to_string(), "Bearer t".to_string())].into(),
            allowed_tools: None,
        }
    }

    #[tokio::test]
    async fn a_session_lists_and_calls_a_servers_tools_over_json_and_event_streams() {
        for event_stream in [false, true] {
            let (url, seen) = serve(event_stream).await;
            let tools = McpTools::connect(&[config(&url)], &hosts("127.0.0.1"))
                .await
                .unwrap();
            assert_eq!(tools.names().collect::<Vec<_>>(), ["add", "fail"]);

            let mut registry = ToolRegistry::new();
            tools.register(&mut registry).unwrap();
            let add = registry.find("add").unwrap();
            assert_eq!(add.executor(), "mcp");
            assert_eq!(add.description(), "Add two numbers.");
            let context = |arguments: Value| ToolContext {
                session_id: "s".into(),
                thread_id: "t".into(),
                call_id: "c".into(),
                user_text: String::new(),
                arguments,
                cancel: izwi_agent::TurnCancel::new(),
            };
            let output = add.invoke(context(json!({"a": 2, "b": 3}))).await.unwrap();
            assert_eq!(output.text, "5\n[image content]");
            let err = registry
                .find("fail")
                .unwrap()
                .invoke(context(json!({})))
                .await
                .unwrap_err();
            assert_eq!(err.to_string(), "Tool error: it broke");

            let seen = lock(&seen).clone();
            let methods: Vec<_> = seen.iter().map(|call| call["method"].clone()).collect();
            assert_eq!(
                methods,
                [
                    "initialize",
                    "notifications/initialized",
                    "tools/list",
                    "tools/list",
                    "tools/call",
                    "tools/call"
                ]
            );
            // The handshake's session and protocol version follow.
            assert!(seen[0]["session"].is_null() && seen[0]["version"].is_null());
            assert_eq!(seen[2]["session"], "sess-1");
            assert_eq!(seen[2]["version"], "2025-03-26");
            assert_eq!(seen[3]["params"]["cursor"], "page-2");
            assert_eq!(seen[4]["params"]["arguments"], json!({"a": 2, "b": 3}));

            // A registry that already has the name refuses the tool.
            assert!(tools.register(&mut registry).is_err());
        }
    }

    #[tokio::test]
    async fn a_server_is_reached_only_on_an_allowed_host() {
        let (url, seen) = serve(false).await;
        let off = McpTools::connect(&[config(&url)], &McpAccess::default())
            .await
            .unwrap_err();
        assert!(off.contains("MCP servers are off"), "{off}");
        let other = McpTools::connect(&[config(&url)], &hosts("example.com"))
            .await
            .unwrap_err();
        assert!(other.contains("does not allow"), "{other}");
        assert!(lock(&seen).is_empty(), "nothing was sent");

        let port = url.rsplit(':').next().unwrap().trim_end_matches("/mcp");
        for list in ["*", &format!("example.com, 127.0.0.1:{port}")] {
            let narrowed = McpServerConfig {
                allowed_tools: Some(vec!["add".to_string()]),
                ..config(&url)
            };
            let tools = McpTools::connect(&[narrowed], &hosts(list)).await.unwrap();
            assert_eq!(tools.names().collect::<Vec<_>>(), ["add"]);
        }
        let file = McpServerConfig {
            url: Some("file:///etc/passwd".to_string()),
            ..config(&url)
        };
        assert!(McpTools::connect(&[file], &hosts("*")).await.is_err());

        // A wrong token is the server's 401, reported with the label.
        let unauthorized = McpServerConfig {
            headers: Default::default(),
            ..config(&url)
        };
        let err = McpTools::connect(&[unauthorized], &hosts("*"))
            .await
            .unwrap_err();
        assert_eq!(err, "MCP server `calc` initialize: HTTP 401");
    }

    /// A local MCP server as a shell script: one tool, which answers with
    /// its `GREETING` after a notification and a line that is not protocol.
    #[cfg(unix)]
    const STDIO_SERVER: &str = r#"
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"initialize"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{}}}\n' "$id";;
    *'"tools/list"'*)
      printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"greet","description":"Greet.","inputSchema":{"type":"object"}}]}}\n' "$id";;
    *'"quit"'*)
      exit 0;;
    *'"tools/call"'*)
      echo "starting up"
      printf '{"jsonrpc":"2.0","method":"notifications/message"}\n'
      printf '{"jsonrpc":"2.0","id":99,"method":"ping"}\n'
      IFS= read -r pong
      printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s from pid %s"}]}}\n' "$id" "$GREETING" "$$";;
  esac
done
"#;

    #[cfg(unix)]
    #[tokio::test]
    async fn a_configured_local_server_runs_once_and_is_shared_until_it_exits() {
        let file = json!({"mcpServers": {"local": {
            "command": "sh",
            "args": ["-c", STDIO_SERVER],
            "env": {"GREETING": "hello"},
        }}});
        let access = McpAccess {
            // No host is allowed: a configured server needs none.
            allowed: AllowedHosts::default(),
            configured: Arc::new(ConfiguredServers::parse(&file.to_string()).unwrap()),
        };
        assert_eq!(access.configured.list(), [("local", "stdio")]);
        let named = |label: &str| McpServerConfig {
            label: label.to_string(),
            url: None,
            connector_id: None,
            headers: Default::default(),
            allowed_tools: None,
        };

        let first = McpTools::connect(&[named("local")], &access).await.unwrap();
        assert_eq!(first.names().collect::<Vec<_>>(), ["greet"]);
        let greeting = first.call("greet", "{}").await.unwrap();
        assert!(greeting.starts_with("hello from pid "), "{greeting}");
        // A second session talks to the same process.
        let second = McpTools::connect(&[named("local")], &access).await.unwrap();
        assert_eq!(second.call("greet", "").await.unwrap(), greeting);

        // The process exits: the call waiting on it fails, and the next
        // session starts it again.
        let quit = first.call("greet", r#"{"quit": true}"#).await.unwrap_err();
        assert!(quit.contains("the server process exited"), "{quit}");
        let third = McpTools::connect(&[named("local")], &access).await.unwrap();
        let restarted = third.call("greet", "{}").await.unwrap();
        assert!(restarted.starts_with("hello from pid ") && restarted != greeting);

        // What a session may not do with configured servers.
        let unknown = McpTools::connect(&[named("other")], &access)
            .await
            .unwrap_err();
        assert!(
            unknown.contains("not in this server's MCP servers file"),
            "{unknown}"
        );
        let with_headers = McpServerConfig {
            headers: [("Authorization".to_string(), "Bearer t".to_string())].into(),
            ..named("local")
        };
        let refused = McpTools::connect(&[with_headers], &access)
            .await
            .unwrap_err();
        assert!(refused.contains("takes no headers"), "{refused}");
    }

    #[tokio::test]
    async fn a_configured_remote_server_needs_no_allowed_host_and_takes_a_sessions_token() {
        let (url, seen) = serve(false).await;
        let file = json!({"mcpServers": {
            "calc": {"url": url, "headers": {"Authorization": "Bearer t"}},
            "locked": {"url": url},
        }});
        let access = McpAccess {
            allowed: AllowedHosts::default(),
            configured: Arc::new(ConfiguredServers::parse(&file.to_string()).unwrap()),
        };
        assert_eq!(
            access.configured.list(),
            [("calc", "http"), ("locked", "http")]
        );
        let named = |label: &str, token: Option<&str>| McpServerConfig {
            label: label.to_string(),
            url: None,
            connector_id: None,
            headers: token
                .map(|token| ("Authorization".to_string(), format!("Bearer {token}")))
                .into_iter()
                .collect(),
            allowed_tools: Some(vec!["add".to_string()]),
        };
        let tools = McpTools::connect(&[named("calc", None)], &access)
            .await
            .unwrap();
        assert_eq!(tools.call("add", "{}").await.unwrap(), "5\n[image content]");
        // The shared connection is reused: one handshake for two sessions.
        McpTools::connect(&[named("calc", None)], &access)
            .await
            .unwrap();
        let handshakes = |seen: &Arc<Mutex<Vec<Value>>>| {
            lock(seen)
                .iter()
                .filter(|call| call["method"] == "initialize")
                .count()
        };
        assert_eq!(handshakes(&seen), 1);

        // A server without its own token takes the session's.
        let denied = McpTools::connect(&[named("locked", None)], &access)
            .await
            .unwrap_err();
        assert_eq!(denied, "MCP server `locked` initialize: HTTP 401");
        McpTools::connect(&[named("locked", Some("t"))], &access)
            .await
            .unwrap();

        // A connector id names the configured server; the session's label
        // names it in the session.
        let connector = McpServerConfig {
            connector_id: Some("calc".to_string()),
            ..named("sums", None)
        };
        let tools = McpTools::connect(&[connector], &access).await.unwrap();
        assert_eq!(tools.server_of("add"), Some("sums"));
        assert_eq!(tools.tools_of("sums").len(), 1);
    }

    #[test]
    fn a_servers_file_is_checked_when_it_is_read() {
        let parse = |file: Value| ConfiguredServers::parse(&file.to_string());
        assert!(parse(json!({})).unwrap().list().is_empty());
        for bad in [
            json!({"mcpServers": {"a b": {"command": "x"}}}),
            json!({"mcpServers": {"a": {"command": " "}}}),
            json!({"mcpServers": {"a": {"url": "file:///etc/passwd"}}}),
            json!({"mcpServers": {"a": {"args": ["x"]}}}),
        ] {
            assert!(parse(bad.clone()).is_err(), "{bad}");
        }
        assert!(ConfiguredServers::load(Path::new("/nonexistent/mcp.json"))
            .list()
            .is_empty());
    }

    #[test]
    fn an_event_stream_yields_each_events_data_across_chunk_boundaries() {
        let mut stream = EventStream::default();
        assert!(stream.push(b"event: message\r\ndata: {\"a\":").is_empty());
        assert_eq!(
            stream.push(b"1}\r\n\r\ndata: x\ndata: y\n\n"),
            ["{\"a\":1}", "x\ny"]
        );
        assert!(stream.push(b": comment\ndata: tail\n").is_empty());
        assert_eq!(stream.finish().as_deref(), Some("tail"));
    }

    #[test]
    fn a_result_without_text_reads_as_its_structured_content() {
        assert_eq!(
            result_text(&json!({"content": [], "structuredContent": {"n": 1}})),
            r#"{"n":1}"#
        );
        let long = "x".repeat(MAX_RESULT_CHARS + 10);
        let text = result_text(&json!({"content": [{"type": "text", "text": long}]}));
        assert!(text.ends_with("[result cut short]"));
    }
}
