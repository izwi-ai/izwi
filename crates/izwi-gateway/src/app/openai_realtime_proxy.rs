//! OpenAI Realtime pass-through: the gateway's `/v1/realtime` hands each
//! session to an izwi-server that runs it.
//!
//! An OpenAI Realtime session is stateful (VAD, conversation, responses,
//! barge-in) and lives on the izwi-server that serves its models, so the
//! gateway does not orchestrate it across stage workers. It authenticates and
//! admits the session (global admission, the tenant's concurrent-work lease,
//! a session cap), dials an upstream izwi-server with that server's own key,
//! and relays frames both ways until either side closes. The upstream's
//! refusal of the upgrade (an unknown `model`, its own capacity) reaches the
//! client as the same HTTP status.
//!
//! `POST /v1/realtime/client_secrets` and `GET /v1/realtime/capabilities`
//! are relayed to the same upstreams. An izwi-server keeps the `ek_` keys it
//! mints in its own memory, so the gateway remembers which upstream minted
//! each key, and for whom. A browser has no gateway API key: the `ek_` key it
//! presents on `/v1/realtime` authenticates it as the principal that minted
//! the key, and its session goes to the upstream that knows the key, which
//! checks it again and applies the session config bound to it. The memory is
//! this process's own, so behind several gateway replicas a key works only
//! through the replica that minted it.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::{
    body::Bytes,
    extract::{
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
        Extension, RawQuery, State,
    },
    http::{header, HeaderMap, Method, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};
use izwi_hooks::Principal;
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_tungstenite::tungstenite::{
    self, client::IntoClientRequest, http::HeaderValue, protocol::WebSocketConfig,
    Message as UpstreamMessage,
};
use tracing::warn;

use crate::api::request_context::RequestContext;
use crate::gateway::GatewayState;

/// The largest message either side may send: a 15 MiB audio append in base64.
pub const MAX_MESSAGE_BYTES: usize = 24 * 1024 * 1024;
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// The whole of a relayed HTTP request: both routes answer from memory.
const UPSTREAM_HTTP_TIMEOUT: Duration = Duration::from_secs(30);
/// The realtime subprotocol a browser-style client offers.
const REALTIME_SUBPROTOCOL: &str = "realtime";
/// The subprotocol prefix a browser presents an ephemeral key with.
const SUBPROTOCOL_KEY_PREFIX: &str = "openai-insecure-api-key.";
/// Client secrets remembered: an izwi-server keeps as many live keys.
const CLIENT_SECRET_MEMORY: usize = 10_000;

/// The izwi-servers OpenAI Realtime sessions go to.
#[derive(Debug)]
pub struct OpenAiRealtimeUpstreams {
    urls: Vec<String>,
    api_key: Option<String>,
    sessions: Arc<Semaphore>,
    max_sessions: usize,
    next: AtomicUsize,
    client: reqwest::Client,
    client_secrets: Mutex<ClientSecretMemory>,
}

/// A client secret an upstream minted through this gateway.
#[derive(Debug, Clone)]
struct ClientSecret {
    upstream: usize,
    /// Who minted the key: its sessions are theirs.
    principal: Principal,
    expires_at: u64,
}

/// The live client secrets, by the SHA-256 of the key.
#[derive(Debug, Default)]
struct ClientSecretMemory {
    order: VecDeque<[u8; 32]>,
    secrets: HashMap<[u8; 32], ClientSecret>,
}

impl ClientSecretMemory {
    fn remember(&mut self, key: &str, secret: ClientSecret, now: u64) {
        self.order.retain(|digest| {
            let live = self
                .secrets
                .get(digest)
                .is_some_and(|secret| secret.expires_at > now);
            if !live {
                self.secrets.remove(digest);
            }
            live
        });
        let digest = digest(key);
        if self.secrets.insert(digest, secret).is_none() {
            self.order.push_back(digest);
        }
        while self.order.len() > CLIENT_SECRET_MEMORY {
            if let Some(oldest) = self.order.pop_front() {
                self.secrets.remove(&oldest);
            }
        }
    }
}

fn digest(key: &str) -> [u8; 32] {
    Sha256::digest(key.as_bytes()).into()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// The ephemeral key a request presents: an `openai-insecure-api-key.ek_…`
/// subprotocol, or an `ek_…` bearer token, as izwi-server reads it.
fn presented_client_secret(headers: &HeaderMap) -> Option<&str> {
    let offered = headers
        .get_all(header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .find_map(|protocol| protocol.trim().strip_prefix(SUBPROTOCOL_KEY_PREFIX));
    offered
        .or_else(|| crate::gateway_security::bearer_token(headers))
        .filter(|key| key.starts_with("ek_"))
}

impl OpenAiRealtimeUpstreams {
    /// `urls` are izwi-server base URLs (`http(s)://` or `ws(s)://`, no path);
    /// `api_key` is their `IZWI_API_KEY`, when set.
    pub fn new(
        urls: Vec<String>,
        api_key: Option<String>,
        max_sessions: usize,
    ) -> Result<Self, String> {
        let urls = urls
            .into_iter()
            .map(|url| websocket_base(&url))
            .collect::<Result<Vec<_>, _>>()?;
        if urls.is_empty() {
            return Err("at least one OpenAI Realtime upstream is required".to_string());
        }
        if max_sessions == 0 {
            return Err("the OpenAI Realtime session cap must be positive".to_string());
        }
        // reqwest's rustls has no built-in crypto provider in this workspace.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = reqwest::Client::builder()
            .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
            .timeout(UPSTREAM_HTTP_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| format!("OpenAI Realtime upstream client: {err}"))?;
        Ok(Self {
            urls,
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            sessions: Arc::new(Semaphore::new(max_sessions)),
            max_sessions,
            next: AtomicUsize::new(0),
            client,
            client_secrets: Mutex::default(),
        })
    }

    pub fn active_sessions(&self) -> usize {
        self.max_sessions - self.sessions.available_permits()
    }

    /// The next upstream, round robin.
    fn pick(&self) -> usize {
        self.next.fetch_add(1, Ordering::Relaxed) % self.urls.len()
    }

    /// The live client secret `headers` present, when this gateway relayed
    /// its minting.
    fn client_secret<'a>(&self, headers: &'a HeaderMap) -> Option<(&'a str, ClientSecret)> {
        let key = presented_client_secret(headers)?;
        let secret = self
            .client_secrets
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .secrets
            .get(&digest(key))
            .filter(|secret| secret.expires_at > unix_now())
            .cloned()?;
        Some((key, secret))
    }

    /// Who a request without a gateway key is, when it opens `/v1/realtime`
    /// with a client secret minted through this gateway.
    pub fn client_secret_principal(&self, headers: &HeaderMap) -> Option<Principal> {
        self.client_secret(headers)
            .map(|(_, secret)| secret.principal)
    }

    /// Relay one small HTTP request to `upstream` and read its answer whole.
    async fn forward(
        &self,
        upstream: usize,
        method: Method,
        path: &str,
        headers: &HeaderMap,
        correlation_id: &str,
        body: Bytes,
    ) -> reqwest::Result<UpstreamAnswer> {
        let base = &self.urls[upstream];
        // Upstreams are kept as WebSocket bases: ws(s) maps onto http(s).
        let url = format!("http{}{path}", base.trim_start_matches("ws"));
        let mut request = self
            .client
            .request(method, url)
            .header("x-request-id", correlation_id)
            .body(body);
        if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
            request = request.header(header::CONTENT_TYPE, content_type.clone());
        }
        if let Some(key) = self.api_key.as_deref() {
            request = request.bearer_auth(key);
        }
        let response = request.send().await?;
        Ok(UpstreamAnswer {
            status: StatusCode::from_u16(response.status().as_u16())
                .unwrap_or(StatusCode::BAD_GATEWAY),
            content_type: response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| axum::http::HeaderValue::from_bytes(value.as_bytes()).ok()),
            body: response.bytes().await?,
        })
    }
}

/// An upstream's answer to a relayed HTTP request.
struct UpstreamAnswer {
    status: StatusCode,
    content_type: Option<axum::http::HeaderValue>,
    body: Bytes,
}

impl IntoResponse for UpstreamAnswer {
    fn into_response(self) -> Response {
        let mut response = (self.status, self.body).into_response();
        if let Some(content_type) = self.content_type {
            response
                .headers_mut()
                .insert(header::CONTENT_TYPE, content_type);
        }
        response
    }
}

/// The 502 for an upstream that could not be reached.
fn upstream_unavailable(error: reqwest::Error) -> Response {
    warn!("realtime upstream request failed: {error}");
    json_error(
        StatusCode::BAD_GATEWAY,
        "The realtime upstream is unavailable.",
    )
}

/// `POST /v1/realtime/client_secrets`: an upstream mints the key, and the
/// gateway remembers which, so the key's sessions reach it.
pub async fn proxy_client_secret(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(upstreams) = state.openai_realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let upstream = upstreams.pick();
    let answer = match upstreams
        .forward(
            upstream,
            Method::POST,
            "/v1/realtime/client_secrets",
            &headers,
            &context.correlation_id,
            body,
        )
        .await
    {
        Ok(answer) => answer,
        Err(error) => return upstream_unavailable(error),
    };
    if answer.status.is_success() {
        let minted = serde_json::from_slice::<serde_json::Value>(&answer.body).unwrap_or_default();
        if let (Some(key), Some(expires_at)) =
            (minted["value"].as_str(), minted["expires_at"].as_u64())
        {
            upstreams
                .client_secrets
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remember(
                    key,
                    ClientSecret {
                        upstream,
                        principal: context.principal.clone(),
                        expires_at,
                    },
                    unix_now(),
                );
        }
    }
    answer.into_response()
}

/// `GET /v1/realtime/capabilities`, as an upstream answers it.
pub async fn proxy_capabilities(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    headers: HeaderMap,
) -> Response {
    let Some(upstreams) = state.openai_realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match upstreams
        .forward(
            upstreams.pick(),
            Method::GET,
            "/v1/realtime/capabilities",
            &headers,
            &context.correlation_id,
            Bytes::new(),
        )
        .await
    {
        Ok(answer) => answer.into_response(),
        Err(error) => upstream_unavailable(error),
    }
}

/// `url` as a `ws(s)://host[:port]` base.
pub(crate) fn websocket_base(url: &str) -> Result<String, String> {
    let url = url.trim().trim_end_matches('/');
    let base = if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else if url.starts_with("ws://") || url.starts_with("wss://") {
        url.to_string()
    } else {
        return Err(format!(
            "OpenAI Realtime upstream `{url}` is not an http(s) or ws(s) URL"
        ));
    };
    let path_start = base.find("://").map_or(0, |index| index + 3);
    if base[path_start..].contains('/') {
        return Err(format!(
            "OpenAI Realtime upstream `{url}` must be a base URL without a path"
        ));
    }
    Ok(base)
}

fn offers_realtime(headers: &HeaderMap) -> bool {
    headers
        .get_all(axum::http::header::SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|protocol| protocol.trim() == REALTIME_SUBPROTOCOL)
}

pub(crate) fn json_error(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(serde_json::json!({"error": {"message": message, "type": "server_error", "code": null}})),
    )
        .into_response()
}

/// `GET /v1/realtime`.
pub async fn proxy_socket(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    RawQuery(query): RawQuery,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(upstreams) = state.openai_realtime.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let Ok(session_permit) = upstreams.sessions.clone().try_acquire_owned() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "The gateway's realtime session capacity is exhausted; retry shortly.",
        );
    };
    let tenant_lease = match state.begin_tenant_work(&context) {
        Ok(lease) => lease,
        Err(error) => return error.into_response(),
    };
    let offered = offers_realtime(&headers);
    // A session opened with a client secret goes to the upstream that minted
    // it, with the secret as its credential there.
    let client_secret = upstreams.client_secret(&headers);
    let upstream = match &client_secret {
        Some((_, secret)) => secret.upstream,
        None => upstreams.pick(),
    };
    let url = format!(
        "{}/v1/realtime{}",
        upstreams.urls[upstream],
        query.map(|query| format!("?{query}")).unwrap_or_default()
    );
    let mut request = match url.into_client_request() {
        Ok(request) => request,
        Err(error) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                &format!("Invalid realtime request: {error}"),
            )
        }
    };
    let upstream_key = client_secret
        .as_ref()
        .map(|(key, _)| *key)
        .or(upstreams.api_key.as_deref());
    if let Some(key) = upstream_key {
        match HeaderValue::from_str(&format!("Bearer {key}")) {
            Ok(value) => {
                request.headers_mut().insert("authorization", value);
            }
            Err(_) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "The upstream API key is not a valid header",
                )
            }
        }
    }
    if offered {
        request.headers_mut().insert(
            "sec-websocket-protocol",
            HeaderValue::from_static(REALTIME_SUBPROTOCOL),
        );
    }
    if let Ok(value) = HeaderValue::from_str(&context.correlation_id) {
        request.headers_mut().insert("x-request-id", value);
    }
    let config = WebSocketConfig::default()
        .max_message_size(Some(MAX_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_MESSAGE_BYTES));
    let connected = tokio::time::timeout(
        UPSTREAM_CONNECT_TIMEOUT,
        tokio_tungstenite::connect_async_with_config(request, Some(config), false),
    )
    .await;
    let (upstream, upstream_response) = match connected {
        Ok(Ok(connected)) => connected,
        Ok(Err(tungstenite::Error::Http(response))) => {
            let status =
                StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = response.into_body().unwrap_or_default();
            return (
                status,
                [(axum::http::header::CONTENT_TYPE, "application/json")],
                body,
            )
                .into_response();
        }
        Ok(Err(error)) => {
            warn!("realtime upstream connection failed: {error}");
            return json_error(
                StatusCode::BAD_GATEWAY,
                "The realtime upstream is unavailable.",
            );
        }
        Err(_) => {
            return json_error(
                StatusCode::GATEWAY_TIMEOUT,
                "The realtime upstream did not answer in time.",
            )
        }
    };
    let profile = upstream_response
        .headers()
        .get("x-izwi-realtime-profile")
        .cloned();
    let upgrade = upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES);
    let upgrade = if offered {
        upgrade.protocols([REALTIME_SUBPROTOCOL])
    } else {
        upgrade
    };
    let mut response = upgrade.on_upgrade(move |client| async move {
        relay(client, upstream, session_permit).await;
        drop(tenant_lease);
    });
    if let Some(profile) = profile {
        response
            .headers_mut()
            .insert("x-izwi-realtime-profile", profile);
    }
    response
}

pub(crate) type Upstream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Relay frames both ways until either side closes or fails.
pub(crate) async fn relay(client: WebSocket, upstream: Upstream, _permit: OwnedSemaphorePermit) {
    let (mut client_tx, mut client_rx) = client.split();
    let (mut upstream_tx, mut upstream_rx) = upstream.split();
    let to_upstream = async {
        while let Some(Ok(message)) = client_rx.next().await {
            let (message, closing) = match message {
                Message::Text(text) => (UpstreamMessage::Text(text.as_str().into()), false),
                Message::Binary(data) => (UpstreamMessage::Binary(data), false),
                Message::Ping(data) => (UpstreamMessage::Ping(data), false),
                Message::Pong(data) => (UpstreamMessage::Pong(data), false),
                Message::Close(frame) => (
                    UpstreamMessage::Close(frame.map(|frame| tungstenite::protocol::CloseFrame {
                        code: frame.code.into(),
                        reason: frame.reason.as_str().into(),
                    })),
                    true,
                ),
            };
            if upstream_tx.send(message).await.is_err() || closing {
                break;
            }
        }
        let _ = upstream_tx.close().await;
    };
    let to_client = async {
        while let Some(Ok(message)) = upstream_rx.next().await {
            let (message, closing) = match message {
                UpstreamMessage::Text(text) => (Message::Text(text.as_str().into()), false),
                UpstreamMessage::Binary(data) => (Message::Binary(data), false),
                UpstreamMessage::Ping(data) => (Message::Ping(data), false),
                UpstreamMessage::Pong(data) => (Message::Pong(data), false),
                UpstreamMessage::Close(frame) => (
                    Message::Close(frame.map(|frame| CloseFrame {
                        code: frame.code.into(),
                        reason: frame.reason.as_str().into(),
                    })),
                    true,
                ),
                UpstreamMessage::Frame(_) => continue,
            };
            if client_tx.send(message).await.is_err() || closing {
                break;
            }
        }
        let _ = client_tx.close().await;
    };
    tokio::select! {
        () = to_upstream => {},
        () = to_client => {},
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_urls_become_websocket_bases() {
        assert_eq!(
            websocket_base("http://10.0.0.5:8080/").unwrap(),
            "ws://10.0.0.5:8080"
        );
        assert_eq!(
            websocket_base("https://voice.example").unwrap(),
            "wss://voice.example"
        );
        assert_eq!(websocket_base("ws://a:1").unwrap(), "ws://a:1");
        assert!(websocket_base("10.0.0.5:8080").is_err());
        assert!(
            websocket_base("http://a/v1").is_err(),
            "a base URL, not a path"
        );
        assert!(OpenAiRealtimeUpstreams::new(vec![], None, 4).is_err());
        let upstreams = OpenAiRealtimeUpstreams::new(
            vec!["http://a:1".into(), "http://b:2".into()],
            Some(" ".into()),
            4,
        )
        .unwrap();
        assert_eq!(upstreams.api_key, None, "a blank key is no key");
        assert_eq!(
            [upstreams.pick(), upstreams.pick(), upstreams.pick()],
            [0, 1, 0]
        );
    }

    #[test]
    fn a_client_secret_is_remembered_until_it_expires() {
        let upstreams =
            OpenAiRealtimeUpstreams::new(vec!["http://a:1".into(), "http://b:2".into()], None, 4)
                .unwrap();
        let secret = |upstream, expires_at| ClientSecret {
            upstream,
            principal: Principal::local_anonymous(),
            expires_at,
        };
        let now = unix_now();
        {
            let mut memory = upstreams.client_secrets.lock().unwrap();
            memory.remember("ek_live", secret(1, now + 600), now);
            memory.remember("ek_expired", secret(0, now), now);
        }
        let presenting = |name: &'static str, value: &str| {
            let mut headers = HeaderMap::new();
            headers.insert(name, value.parse().unwrap());
            headers
        };
        // A browser offers the key as a subprotocol; a server-side client
        // may send it as a bearer token.
        for headers in [
            presenting(
                "sec-websocket-protocol",
                "realtime, openai-insecure-api-key.ek_live",
            ),
            presenting("authorization", "Bearer ek_live"),
        ] {
            let (key, secret) = upstreams.client_secret(&headers).expect("a live secret");
            assert_eq!((key, secret.upstream), ("ek_live", 1));
            assert!(upstreams.client_secret_principal(&headers).is_some());
        }
        for headers in [
            presenting("authorization", "Bearer ek_expired"),
            presenting("authorization", "Bearer ek_unknown"),
            presenting("authorization", "Bearer not-an-ephemeral-key"),
            presenting("sec-websocket-protocol", "realtime"),
        ] {
            assert!(upstreams.client_secret(&headers).is_none());
        }
        // The next key minted drops the expired one from the memory.
        let mut memory = upstreams.client_secrets.lock().unwrap();
        memory.remember("ek_next", secret(0, now + 600), now);
        assert_eq!(memory.secrets.len(), 2);
        assert_eq!(memory.order.len(), 2);
    }
}
