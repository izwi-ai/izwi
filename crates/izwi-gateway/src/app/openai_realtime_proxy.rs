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

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade},
        Extension, RawQuery, State,
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};
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
/// The realtime subprotocol a browser-style client offers.
const REALTIME_SUBPROTOCOL: &str = "realtime";

/// The izwi-servers OpenAI Realtime sessions go to.
#[derive(Debug)]
pub struct OpenAiRealtimeUpstreams {
    urls: Vec<String>,
    api_key: Option<String>,
    sessions: Arc<Semaphore>,
    max_sessions: usize,
    next: AtomicUsize,
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
        Ok(Self {
            urls,
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            sessions: Arc::new(Semaphore::new(max_sessions)),
            max_sessions,
            next: AtomicUsize::new(0),
        })
    }

    pub fn active_sessions(&self) -> usize {
        self.max_sessions - self.sessions.available_permits()
    }

    /// The next upstream, round robin.
    fn pick(&self) -> &str {
        let index = self.next.fetch_add(1, Ordering::Relaxed) % self.urls.len();
        &self.urls[index]
    }
}

/// `url` as a `ws(s)://host[:port]` base.
fn websocket_base(url: &str) -> Result<String, String> {
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

fn json_error(status: StatusCode, message: &str) -> Response {
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
    let url = format!(
        "{}/v1/realtime{}",
        upstreams.pick(),
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
    if let Some(key) = upstreams.api_key.as_deref() {
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

type Upstream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Relay frames both ways until either side closes or fails.
async fn relay(client: WebSocket, upstream: Upstream, _permit: OwnedSemaphorePermit) {
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
            ["ws://a:1", "ws://b:2", "ws://a:1"]
        );
    }
}
