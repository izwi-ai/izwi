//! Speech API pass-through: the gateway serves `/v1/audio/*` by relaying
//! each request to an izwi-server upstream, which runs the models, encodes
//! the audio and keeps any state.
//!
//! The gateway authenticates and admits the request (the inference
//! middleware's global admission and tenant lease), then relays it whole to
//! one of `IZWI_GATEWAY_AUDIO_UPSTREAMS`, round robin: request and response
//! bodies stream through unbuffered, so chunked `audio` streams, SSE and
//! large uploads behave as they do on the server. `/v1/audio/speech/stream`
//! is relayed as a WebSocket, as `/v1/realtime` is.
//!
//! `GET /v1/audio/speech/{id}` must reach the server that ran the request:
//! the gateway remembers which upstream answered each `X-Izwi-Speech-Id`, and
//! asks every upstream in turn for an id it has not seen (a batch item's).
//!
//! Voices and consents (`/v1/audio/voices`, `/v1/audio/voice_consents`) are
//! relayed only with `IZWI_GATEWAY_AUDIO_SHARED_STATE=true`, when every
//! upstream shares one records database and media store; otherwise a voice
//! made through one upstream would be missing on the next.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{ws::WebSocketUpgrade, Extension, RawQuery, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
};
use futures::TryStreamExt;
use tokio_tungstenite::tungstenite::{self, client::IntoClientRequest, protocol::WebSocketConfig};
use tracing::warn;

use crate::api::request_context::RequestContext;
use crate::app::openai_realtime_proxy::{json_error, relay, websocket_base, MAX_MESSAGE_BYTES};
use crate::gateway::GatewayState;

const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Speech ids remembered for lookup routing.
const SPEECH_ID_MEMORY: usize = 8_192;

/// Request headers relayed upstream (the gateway's own credentials are not).
const FORWARDED_REQUEST_HEADERS: &[&str] = &[
    "content-type",
    "accept",
    "accept-encoding",
    "x-request-id",
    "idempotency-key",
    "x-izwi-service-tier",
];
/// Response headers that describe the hop, not the content.
const HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-connection",
    "te",
    "trailer",
];

/// The izwi-servers speech requests go to.
pub struct AudioUpstreams {
    bases: Vec<String>,
    api_key: Option<String>,
    shared_state: bool,
    client: reqwest::Client,
    next: AtomicUsize,
    speech_ids: Mutex<SpeechIdMemory>,
    sessions: std::sync::Arc<tokio::sync::Semaphore>,
}

impl std::fmt::Debug for AudioUpstreams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioUpstreams")
            .field("bases", &self.bases)
            .field("shared_state", &self.shared_state)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct SpeechIdMemory {
    order: VecDeque<String>,
    upstream: HashMap<String, usize>,
}

impl SpeechIdMemory {
    fn remember(&mut self, id: String, upstream: usize) {
        if self.upstream.insert(id.clone(), upstream).is_none() {
            self.order.push_back(id);
        }
        while self.order.len() > SPEECH_ID_MEMORY {
            if let Some(oldest) = self.order.pop_front() {
                self.upstream.remove(&oldest);
            }
        }
    }
}

/// `url` as an `http(s)://host[:port]` base.
fn http_base(url: &str) -> Result<String, String> {
    let url = url.trim().trim_end_matches('/');
    let base = if let Some(rest) = url.strip_prefix("wss://") {
        format!("https://{rest}")
    } else if let Some(rest) = url.strip_prefix("ws://") {
        format!("http://{rest}")
    } else if url.starts_with("http://") || url.starts_with("https://") {
        url.to_string()
    } else {
        return Err(format!("Audio upstream `{url}` is not an http(s) URL"));
    };
    let path_start = base.find("://").map_or(0, |index| index + 3);
    if base[path_start..].contains('/') {
        return Err(format!(
            "Audio upstream `{url}` must be a base URL without a path"
        ));
    }
    Ok(base)
}

impl AudioUpstreams {
    pub fn new(
        urls: Vec<String>,
        api_key: Option<String>,
        shared_state: bool,
        max_sessions: usize,
    ) -> Result<Self, String> {
        let bases = urls
            .iter()
            .map(|url| http_base(url))
            .collect::<Result<Vec<_>, _>>()?;
        if bases.is_empty() {
            return Err("at least one audio upstream is required".to_string());
        }
        // reqwest's rustls has no built-in crypto provider in this workspace.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = reqwest::Client::builder()
            .connect_timeout(UPSTREAM_CONNECT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|err| format!("audio upstream client: {err}"))?;
        Ok(Self {
            bases,
            api_key: api_key.filter(|key| !key.trim().is_empty()),
            shared_state,
            client,
            next: AtomicUsize::new(0),
            speech_ids: Mutex::new(SpeechIdMemory::default()),
            sessions: std::sync::Arc::new(tokio::sync::Semaphore::new(max_sessions.max(1))),
        })
    }

    pub fn shared_state(&self) -> bool {
        self.shared_state
    }

    fn pick(&self) -> usize {
        self.next.fetch_add(1, Ordering::Relaxed) % self.bases.len()
    }

    fn remember(&self, id: &str, upstream: usize) {
        self.speech_ids
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remember(id.to_string(), upstream);
    }

    fn remembered(&self, id: &str) -> Option<usize> {
        self.speech_ids
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .upstream
            .get(id)
            .copied()
    }
}

fn upstreams(state: &GatewayState) -> Option<std::sync::Arc<AudioUpstreams>> {
    state.audio_upstreams.clone()
}

/// Relays one HTTP request to `upstream`, streaming both bodies.
async fn forward(
    upstreams: &AudioUpstreams,
    upstream: usize,
    method: Method,
    path_and_query: &str,
    headers: &HeaderMap,
    correlation_id: &str,
    body: reqwest::Body,
) -> reqwest::Result<reqwest::Response> {
    let url = format!("{}{}", upstreams.bases[upstream], path_and_query);
    let mut request = upstreams.client.request(method, url).body(body);
    for name in FORWARDED_REQUEST_HEADERS {
        if let Some(value) = headers.get(*name) {
            request = request.header(*name, value.clone());
        }
    }
    if !headers.contains_key("x-request-id") {
        request = request.header("x-request-id", correlation_id);
    }
    if let Some(key) = upstreams.api_key.as_deref() {
        request = request.bearer_auth(key);
    }
    request.send().await
}

/// The 502 for an upstream that could not be reached.
fn upstream_unavailable(error: reqwest::Error) -> Response {
    warn!("audio upstream request failed: {error}");
    json_error(
        StatusCode::BAD_GATEWAY,
        "The audio upstream is unavailable.",
    )
}

/// The upstream's response, streamed back.
fn relay_response(response: reqwest::Response) -> Response {
    let status =
        StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);
    for (name, value) in response.headers() {
        if HOP_HEADERS.contains(&name.as_str()) {
            continue;
        }
        if let (Ok(name), Ok(value)) = (
            HeaderName::from_bytes(name.as_str().as_bytes()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(name, value);
        }
    }
    let stream = response.bytes_stream().map_err(std::io::Error::other);
    builder
        .body(Body::from_stream(stream))
        .unwrap_or_else(|_| StatusCode::BAD_GATEWAY.into_response())
}

/// `POST /v1/audio/{speech,speech/batch,transcriptions,translations}` and,
/// with shared state, the voice and consent routes.
pub async fn proxy_http(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    request: Request,
) -> Response {
    let Some(upstreams) = upstreams(&state) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let path_and_query = original_path_and_query(&request);
    if is_shared_state_route(&path_and_query) && !upstreams.shared_state {
        return json_error(
            StatusCode::NOT_FOUND,
            "Voices and consents are served by the gateway only when its upstreams share one records database and media store (IZWI_GATEWAY_AUDIO_SHARED_STATE).",
        );
    }
    let (parts, body) = request.into_parts();
    let upstream = upstreams.pick();
    let body = reqwest::Body::wrap_stream(body.into_data_stream());
    let response = match forward(
        &upstreams,
        upstream,
        parts.method,
        &path_and_query,
        &parts.headers,
        &context.correlation_id,
        body,
    )
    .await
    {
        Ok(response) => response,
        Err(error) => return upstream_unavailable(error),
    };
    if let Some(id) = response
        .headers()
        .get("x-izwi-speech-id")
        .and_then(|value| value.to_str().ok())
    {
        upstreams.remember(id, upstream);
    }
    relay_response(response)
}

/// The request's path and query as the client sent them: the `/v1` nest
/// strips its prefix from the URI handlers see.
fn original_path_and_query(request: &Request) -> String {
    let uri = request
        .extensions()
        .get::<axum::extract::OriginalUri>()
        .map(|original| &original.0)
        .unwrap_or(request.uri());
    uri.path_and_query()
        .map(|value| value.as_str().to_string())
        .unwrap_or_default()
}

pub(crate) fn is_shared_state_route(path: &str) -> bool {
    path.starts_with("/v1/audio/voices")
        || path.starts_with("/v1/audio/voice_consents")
        || path.starts_with("/v1/audio/consent_phrases")
}

/// `GET /v1/audio/speech/{id}`: the upstream that served the id, or each in
/// turn until one knows it.
pub async fn proxy_speech_lookup(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    request: Request,
) -> Response {
    let Some(upstreams) = upstreams(&state) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let path = original_path_and_query(&request);
    let path = path.split('?').next().unwrap_or_default().to_string();
    let id = path.rsplit('/').next().unwrap_or_default().to_string();
    let order: Vec<usize> = match upstreams.remembered(&id) {
        Some(upstream) => vec![upstream],
        None => (0..upstreams.bases.len()).collect(),
    };
    let mut last = None;
    for upstream in order {
        match forward(
            &upstreams,
            upstream,
            Method::GET,
            &path,
            request.headers(),
            &context.correlation_id,
            reqwest::Body::from(Vec::new()),
        )
        .await
        {
            Ok(response) if response.status() == reqwest::StatusCode::NOT_FOUND => {
                last = Some(response);
            }
            Ok(response) => {
                upstreams.remember(&id, upstream);
                return relay_response(response);
            }
            Err(error) => return upstream_unavailable(error),
        }
    }
    match last {
        Some(response) => relay_response(response),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

/// `GET /v1/audio/speech/stream`: the text-in speech socket, relayed whole.
pub async fn proxy_speech_stream(
    State(state): State<GatewayState>,
    Extension(context): Extension<RequestContext>,
    RawQuery(query): RawQuery,
    upgrade: WebSocketUpgrade,
) -> Response {
    let Some(upstreams) = upstreams(&state) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let query = query.map(|query| format!("?{query}")).unwrap_or_default();
    let Ok(permit) = upstreams.sessions.clone().try_acquire_owned() else {
        return json_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "The gateway's speech stream capacity is exhausted; retry shortly.",
        );
    };
    let tenant_lease = match state.begin_tenant_work(&context) {
        Ok(lease) => lease,
        Err(error) => return error.into_response(),
    };
    let base = match websocket_base(&upstreams.bases[upstreams.pick()]) {
        Ok(base) => base,
        Err(error) => return json_error(StatusCode::INTERNAL_SERVER_ERROR, &error),
    };
    let mut upstream_request =
        match format!("{base}/v1/audio/speech/stream{query}").into_client_request() {
            Ok(request) => request,
            Err(error) => {
                return json_error(
                    StatusCode::BAD_REQUEST,
                    &format!("Invalid request: {error}"),
                )
            }
        };
    if let Some(key) = upstreams.api_key.as_deref() {
        if let Ok(value) = HeaderValue::from_str(&format!("Bearer {key}")) {
            upstream_request
                .headers_mut()
                .insert("authorization", value);
        }
    }
    if let Ok(value) = HeaderValue::from_str(&context.correlation_id) {
        upstream_request.headers_mut().insert("x-request-id", value);
    }
    let config = WebSocketConfig::default()
        .max_message_size(Some(MAX_MESSAGE_BYTES))
        .max_frame_size(Some(MAX_MESSAGE_BYTES));
    let connected = tokio::time::timeout(
        UPSTREAM_CONNECT_TIMEOUT,
        tokio_tungstenite::connect_async_with_config(upstream_request, Some(config), false),
    )
    .await;
    let upstream = match connected {
        Ok(Ok((upstream, _))) => upstream,
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
            warn!("speech stream upstream connection failed: {error}");
            return json_error(
                StatusCode::BAD_GATEWAY,
                "The audio upstream is unavailable.",
            );
        }
        Err(_) => {
            return json_error(
                StatusCode::GATEWAY_TIMEOUT,
                "The audio upstream did not answer in time.",
            )
        }
    };
    upgrade
        .max_message_size(MAX_MESSAGE_BYTES)
        .max_frame_size(MAX_MESSAGE_BYTES)
        .on_upgrade(move |client| async move {
            relay(client, upstream, permit).await;
            drop(tenant_lease);
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstreams_are_http_bases_picked_in_turn() {
        assert_eq!(http_base("ws://a:1").unwrap(), "http://a:1");
        assert_eq!(http_base("https://b/").unwrap(), "https://b");
        assert!(http_base("b:1").is_err());
        assert!(http_base("http://b/v1").is_err());
        let upstreams = AudioUpstreams::new(
            vec!["http://a:1".into(), "http://b:2".into()],
            Some(" ".into()),
            false,
            4,
        )
        .unwrap();
        assert!(upstreams.api_key.is_none());
        assert_eq!(
            [upstreams.pick(), upstreams.pick(), upstreams.pick()],
            [0, 1, 0]
        );
        upstreams.remember("speech_x", 1);
        assert_eq!(upstreams.remembered("speech_x"), Some(1));
        assert!(AudioUpstreams::new(Vec::new(), None, false, 4).is_err());
    }

    #[test]
    fn voices_need_shared_state() {
        assert!(is_shared_state_route("/v1/audio/voices"));
        assert!(is_shared_state_route("/v1/audio/voice_consents/cons_1"));
        assert!(!is_shared_state_route("/v1/audio/speech"));
        let mut memory = SpeechIdMemory::default();
        for index in 0..SPEECH_ID_MEMORY + 5 {
            memory.remember(format!("speech_{index}"), 0);
        }
        assert_eq!(memory.order.len(), SPEECH_ID_MEMORY);
        assert!(!memory.upstream.contains_key("speech_0"));
    }
}
