//! A Model Context Protocol client for agent turns: the tools of a remote
//! MCP server (Streamable HTTP transport) as tools a reply may call.
//!
//! A session names its servers; the server connects to one only when its
//! host is in `IZWI_MCP_ALLOWED_HOSTS`, so API clients cannot point the
//! server at arbitrary addresses on its network. The client speaks the
//! request/response subset an agent needs: `initialize`, `tools/list` and
//! `tools/call`. A response may arrive as JSON or as a server-sent event
//! stream; requests the MCP server sends to the client are not answered.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use izwi_agent::{AgentError, Tool, ToolContext, ToolOutput, ToolRegistry};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde::Deserialize;
use serde_json::{json, Value};

/// The protocol revision this client speaks.
const PROTOCOL_VERSION: &str = "2025-06-18";
pub(crate) const ALLOWED_HOSTS_ENV: &str = "IZWI_MCP_ALLOWED_HOSTS";
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
    #[serde(alias = "server_url")]
    pub(crate) url: String,
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

    /// The URL of `config`, when the server may connect to it.
    fn check(&self, config: &McpServerConfig) -> Result<reqwest::Url, String> {
        let label = &config.label;
        if self.0.is_empty() {
            return Err(format!(
                "MCP servers are off: set {ALLOWED_HOSTS_ENV} to the hosts this server may connect to"
            ));
        }
        let url = reqwest::Url::parse(&config.url)
            .map_err(|err| format!("MCP server `{label}` has an invalid URL: {err}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!("MCP server `{label}` must use http or https"));
        }
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
    url: reqwest::Url,
    http: reqwest::Client,
    headers: HeaderMap,
    session_id: Mutex<Option<String>>,
    protocol_version: Mutex<String>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Headers may carry credentials.
        f.debug_struct("McpClient")
            .field("label", &self.label)
            .field("url", &self.url.as_str())
            .finish_non_exhaustive()
    }
}

impl McpClient {
    /// Connect to `config`'s server and complete the protocol handshake.
    pub(crate) async fn connect(
        config: &McpServerConfig,
        allowed: &AllowedHosts,
    ) -> Result<Arc<Self>, String> {
        let label = config.label.trim();
        if label.is_empty()
            || !label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "MCP server label {label:?} must be letters, digits, `_` or `-`"
            ));
        }
        let url = allowed.check(config)?;
        let mut headers = HeaderMap::new();
        for (name, value) in &config.headers {
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
            url,
            http,
            headers,
            session_id: Mutex::new(None),
            protocol_version: Mutex::new(PROTOCOL_VERSION.to_string()),
            next_id: AtomicU64::new(1),
        });

        let initialized = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "izwi", "version": env!("CARGO_PKG_VERSION")},
                }),
            )
            .await?;
        if let Some(version) = initialized["protocolVersion"].as_str() {
            *lock(&client.protocol_version) = version.to_string();
        }
        client.notify("notifications/initialized").await?;
        Ok(client)
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
        let response = self
            .post(&body, method != "initialize")
            .await
            .map_err(|err| failed(err.to_string()))?;
        if !response.status().is_success() {
            return Err(failed(format!("HTTP {}", response.status().as_u16())));
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
        let message = if event_stream {
            read_event_stream(response, id).await.map_err(failed)?
        } else {
            let bytes = read_capped(response).await.map_err(failed)?;
            serde_json::from_slice::<Value>(&bytes)
                .map_err(|err| failed(format!("invalid JSON: {err}")))?
        };
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
        let response = self
            .post(&json!({"jsonrpc": "2.0", "method": method}), true)
            .await
            .map_err(|err| format!("MCP server `{}` {method}: {err}", self.label))?;
        if !response.status().is_success() {
            return Err(format!(
                "MCP server `{}` {method}: HTTP {}",
                self.label,
                response.status().as_u16()
            ));
        }
        Ok(())
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
    tools: Vec<(Arc<McpClient>, McpToolInfo)>,
}

impl PartialEq for McpTools {
    fn eq(&self, other: &Self) -> bool {
        self.tools.len() == other.tools.len()
            && self
                .tools
                .iter()
                .zip(&other.tools)
                .all(|(a, b)| Arc::ptr_eq(&a.0, &b.0) && a.1 == b.1)
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
        allowed: &AllowedHosts,
    ) -> Result<Self, String> {
        if servers.len() > MAX_SERVERS {
            return Err(format!(
                "A session names at most {MAX_SERVERS} MCP servers, not {}",
                servers.len()
            ));
        }
        let mut tools: Vec<(Arc<McpClient>, McpToolInfo)> = Vec::new();
        for config in servers {
            let client = McpClient::connect(config, allowed).await?;
            for info in client.list_tools(config.allowed_tools.as_deref()).await? {
                if tools.iter().any(|(_, other)| other.name == info.name) {
                    return Err(format!(
                        "MCP tool `{}` is offered twice; narrow one server with `allowed_tools`",
                        info.name
                    ));
                }
                tools.push((client.clone(), info));
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
            .map(|(_, info)| {
                izwi_agent::function_definition(
                    &info.name,
                    &info.description,
                    info.input_schema.clone(),
                )
            })
            .collect()
    }

    /// The label of the server that offers `name`.
    pub(crate) fn server_of(&self, name: &str) -> Option<&str> {
        self.tools
            .iter()
            .find(|(_, info)| info.name == name)
            .map(|(client, _)| client.label.as_str())
    }

    /// The tools of the server labelled `label`.
    pub(crate) fn tools_of(&self, label: &str) -> Vec<&McpToolInfo> {
        self.tools
            .iter()
            .filter(|(client, _)| client.label == label)
            .map(|(_, info)| info)
            .collect()
    }

    /// Call `name` with `arguments` (a JSON object as text).
    pub(crate) async fn call(&self, name: &str, arguments: &str) -> Result<String, String> {
        let Some((client, _)) = self.tools.iter().find(|(_, info)| info.name == name) else {
            return Err(format!("No MCP tool named `{name}` is available."));
        };
        let arguments = match arguments.trim() {
            "" => json!({}),
            arguments => serde_json::from_str::<Value>(arguments)
                .map_err(|err| format!("Invalid arguments: {err}"))?,
        };
        client.call_tool(name, arguments).await
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|(_, info)| info.name.as_str())
    }

    /// Add the tools to `registry`. `Err` names a tool the registry already
    /// has.
    pub(crate) fn register(&self, registry: &mut ToolRegistry) -> Result<(), String> {
        let taken = registry.tool_names();
        for (client, info) in &self.tools {
            if taken.contains(&info.name) {
                return Err(format!(
                    "MCP tool `{}` has the name of another tool of the session",
                    info.name
                ));
            }
            registry.register(McpTool {
                client: client.clone(),
                info: info.clone(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    use axum::routing::post;
    use axum::Router;

    /// A small MCP server: two tools, answering as JSON or as an event
    /// stream, with a session id and a required bearer token.
    async fn serve(event_stream: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
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

    fn config(url: &str) -> McpServerConfig {
        McpServerConfig {
            label: "calc".to_string(),
            url: url.to_string(),
            headers: [("Authorization".to_string(), "Bearer t".to_string())].into(),
            allowed_tools: None,
        }
    }

    #[tokio::test]
    async fn a_session_lists_and_calls_a_servers_tools_over_json_and_event_streams() {
        for event_stream in [false, true] {
            let (url, seen) = serve(event_stream).await;
            let allowed = AllowedHosts::parse("127.0.0.1");
            let tools = McpTools::connect(&[config(&url)], &allowed).await.unwrap();
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
        let off = McpTools::connect(&[config(&url)], &AllowedHosts::default())
            .await
            .unwrap_err();
        assert!(off.contains("MCP servers are off"), "{off}");
        let other = McpTools::connect(&[config(&url)], &AllowedHosts::parse("example.com"))
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
            let tools = McpTools::connect(&[narrowed], &AllowedHosts::parse(list))
                .await
                .unwrap();
            assert_eq!(tools.names().collect::<Vec<_>>(), ["add"]);
        }
        let file = McpServerConfig {
            url: "file:///etc/passwd".to_string(),
            ..config(&url)
        };
        assert!(McpTools::connect(&[file], &AllowedHosts::parse("*"))
            .await
            .is_err());

        // A wrong token is the server's 401, reported with the label.
        let unauthorized = McpServerConfig {
            headers: Default::default(),
            ..config(&url)
        };
        let err = McpTools::connect(&[unauthorized], &AllowedHosts::parse("*"))
            .await
            .unwrap_err();
        assert_eq!(err, "MCP server `calc` initialize: HTTP 401");
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
