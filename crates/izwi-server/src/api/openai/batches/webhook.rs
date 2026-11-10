//! Completion webhooks: a batch created with `metadata.webhook_url` has an
//! event posted to that URL when it ends.
//!
//! ```json
//! {"id": "evt_…", "object": "event", "type": "batch.completed",
//!  "created_at": 1762790400, "data": {"id": "batch_…"}}
//! ```
//!
//! `type` is `batch.completed`, `batch.failed`, `batch.cancelled` or
//! `batch.expired`, as in OpenAI's webhook events, and a delivery is signed
//! as OpenAI's are (Standard Webhooks): `webhook-id`, `webhook-timestamp`
//! and `webhook-signature: v1,<base64 HMAC-SHA256 of "id.timestamp.body">`.
//!
//! The server posts to a URL a client chose, so the feature is off until
//! the operator sets `IZWI_BATCH_WEBHOOK_SECRET`. A URL must be `https`
//! (`IZWI_BATCH_WEBHOOK_ALLOW_HTTP` adds `http`) and every address of its
//! host must be public (`IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE` lifts that); the
//! connection is pinned to the addresses that were checked, no proxy is
//! used and no redirect is followed. A delivery is tried three times and
//! then dropped. It runs apart from the batch, which has already ended, so
//! the batch object stays the record of what happened.

use std::net::SocketAddr;
use std::time::Duration;

use base64::Engine;
use hmac::{Hmac, KeyInit, Mac};
use izwi_env::EnvSnapshot;
use serde_json::{json, Value};
use sha2::Sha256;
use tracing::{info, warn};

use crate::batch_api_store::{BatchRecord, BatchStatus};
use crate::error::ApiError;

/// The `metadata` key naming a batch's webhook.
pub(crate) const WEBHOOK_URL_KEY: &str = "webhook_url";
const ATTEMPTS: u32 = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// The operator's webhook settings; present only when a secret is set.
#[derive(Clone, PartialEq)]
pub(crate) struct WebhookPolicy {
    pub(crate) secret: String,
    pub(crate) allow_http: bool,
    pub(crate) allow_private: bool,
    /// The wait before the second attempt; the third waits four times it.
    pub(crate) retry_backoff: Duration,
}

impl std::fmt::Debug for WebhookPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookPolicy")
            .field("secret", &"<redacted>")
            .field("allow_http", &self.allow_http)
            .field("allow_private", &self.allow_private)
            .finish()
    }
}

impl WebhookPolicy {
    pub(crate) fn from_env(env: &EnvSnapshot) -> anyhow::Result<Option<Self>> {
        let allow_http = env.flag_or("IZWI_BATCH_WEBHOOK_ALLOW_HTTP", false)?;
        let allow_private = env.flag_or("IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE", false)?;
        Ok(env
            .get("IZWI_BATCH_WEBHOOK_SECRET")?
            .map(str::trim)
            .filter(|secret| !secret.is_empty())
            .map(|secret| Self {
                secret: secret.to_string(),
                allow_http,
                allow_private,
                retry_backoff: Duration::from_secs(2),
            }))
    }

    /// The URL, when it may be posted to. A host given as an address is
    /// checked here; a host name is resolved and checked at delivery.
    fn parse_url(&self, url: &str) -> Result<reqwest::Url, String> {
        let url = reqwest::Url::parse(url.trim()).map_err(|err| format!("invalid URL: {err}"))?;
        match url.scheme() {
            "https" => {}
            "http" if self.allow_http => {}
            "http" => {
                return Err(
                    "it must use https (IZWI_BATCH_WEBHOOK_ALLOW_HTTP allows http)".to_string(),
                )
            }
            _ => return Err("it must use https".to_string()),
        }
        if !url.username().is_empty() || url.password().is_some() {
            return Err("it cannot carry credentials".to_string());
        }
        let host = host_of(&url).ok_or("it has no host")?;
        if let Ok(address) = host.parse::<std::net::IpAddr>() {
            self.check_addresses(&host, &[SocketAddr::new(address, 0)])?;
        }
        Ok(url)
    }

    fn check_addresses(&self, host: &str, addresses: &[SocketAddr]) -> Result<(), String> {
        let public = |address: &SocketAddr| izwi_core::media_policy::is_public_ip(address.ip());
        if addresses.is_empty() || !(self.allow_private || addresses.iter().all(public)) {
            return Err(format!(
                "host {host} is not publicly routable (IZWI_BATCH_WEBHOOK_ALLOW_PRIVATE allows \
                 private addresses)"
            ));
        }
        Ok(())
    }

    /// The signing key: a `whsec_` secret is base64 of the key, as Standard
    /// Webhooks and OpenAI's SDKs read it; any other secret is the key.
    fn key(&self) -> Vec<u8> {
        self.secret
            .strip_prefix("whsec_")
            .and_then(|encoded| {
                base64::engine::general_purpose::STANDARD
                    .decode(encoded)
                    .ok()
            })
            .unwrap_or_else(|| self.secret.as_bytes().to_vec())
    }

    /// The `webhook-signature` of `body` sent as `id` at `timestamp`.
    fn signature(&self, id: &str, timestamp: i64, body: &str) -> String {
        let mut mac =
            Hmac::<Sha256>::new_from_slice(&self.key()).expect("HMAC accepts any key length");
        mac.update(format!("{id}.{timestamp}.{body}").as_bytes());
        format!(
            "v1,{}",
            base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
        )
    }
}

fn host_of(url: &reqwest::Url) -> Option<String> {
    url.host_str()
        .map(|host| host.trim_matches(['[', ']']).to_string())
}

/// Check a new batch's `metadata.webhook_url`: a `400` when webhooks are
/// off or the URL may not be posted to.
pub(crate) fn validate_metadata(
    policy: Option<&WebhookPolicy>,
    metadata: Option<&Value>,
) -> Result<(), ApiError> {
    let Some(url) = metadata.and_then(|metadata| metadata.get(WEBHOOK_URL_KEY)) else {
        return Ok(());
    };
    let refused = |reason: String| {
        ApiError::invalid_param(
            "metadata.webhook_url",
            format!("`metadata.webhook_url` is refused: {reason}"),
        )
    };
    let Some(policy) = policy else {
        return Err(refused(
            "completion webhooks are disabled on this server; the operator enables them with \
             IZWI_BATCH_WEBHOOK_SECRET"
                .to_string(),
        ));
    };
    policy
        .parse_url(url.as_str().unwrap_or_default())
        .map(|_| ())
        .map_err(refused)
}

fn event_type(status: BatchStatus) -> Option<&'static str> {
    match status {
        BatchStatus::Completed => Some("batch.completed"),
        BatchStatus::Failed => Some("batch.failed"),
        BatchStatus::Cancelled => Some("batch.cancelled"),
        BatchStatus::Expired => Some("batch.expired"),
        _ => None,
    }
}

/// Post the event of `batch` having ended as `status`, when the batch
/// names a webhook. Returns at once; the delivery runs on its own task.
pub(crate) fn notify(policy: Option<&WebhookPolicy>, batch: &BatchRecord, status: BatchStatus) {
    let (Some(policy), Some(event_type)) = (policy, event_type(status)) else {
        return;
    };
    let Some(url) = batch
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get(WEBHOOK_URL_KEY))
        .and_then(Value::as_str)
    else {
        return;
    };
    let policy = policy.clone();
    let url = url.to_string();
    let batch_id = batch.id.clone();
    tokio::spawn(async move {
        let event_id = crate::ids::new_prefixed_id("evt_");
        let body = json!({
            "id": event_id,
            "object": "event",
            "type": event_type,
            "created_at": now_secs(),
            "data": {"id": batch_id},
        })
        .to_string();
        match deliver(&policy, &url, &event_id, &body).await {
            Ok(attempt) => {
                info!(
                    batch_id,
                    event_type, attempt, "A batch webhook was delivered"
                )
            }
            Err(reason) => warn!(
                batch_id,
                event_type, reason, "A batch webhook was not delivered"
            ),
        }
    });
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX)
        })
}

/// Post `body` to `url`, up to three times; the attempt that was accepted
/// (a 2xx), or why the last one was not.
pub(crate) async fn deliver(
    policy: &WebhookPolicy,
    url: &str,
    event_id: &str,
    body: &str,
) -> Result<u32, String> {
    // reqwest's rustls has no built-in crypto provider in this workspace.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let url = policy.parse_url(url)?;
    let mut attempt = 1;
    loop {
        match post(policy, &url, event_id, body).await {
            Ok(()) => return Ok(attempt),
            Err(reason) if attempt == ATTEMPTS => return Err(reason),
            Err(_) => {}
        }
        tokio::time::sleep(policy.retry_backoff * 4u32.pow(attempt - 1)).await;
        attempt += 1;
    }
}

/// One attempt: resolve the host, check its addresses, and post to them.
async fn post(
    policy: &WebhookPolicy,
    url: &reqwest::Url,
    event_id: &str,
    body: &str,
) -> Result<(), String> {
    let host = host_of(url).ok_or("the URL has no host")?;
    let port = url.port_or_known_default().ok_or("the URL has no port")?;
    let addresses: Vec<SocketAddr> = match host.parse::<std::net::IpAddr>() {
        Ok(address) => vec![SocketAddr::new(address, port)],
        Err(_) => tokio::time::timeout(
            CONNECT_TIMEOUT,
            tokio::net::lookup_host((host.as_str(), port)),
        )
        .await
        .map_err(|_| format!("resolving {host} timed out"))?
        .map_err(|err| format!("could not resolve {host}: {err}"))?
        .collect(),
    };
    policy.check_addresses(&host, &addresses)?;
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(REQUEST_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        // A proxy would bypass the checked, pinned addresses.
        .no_proxy()
        .resolve_to_addrs(&host, &addresses)
        .build()
        .map_err(|err| format!("HTTP client failed: {err}"))?;
    let timestamp = now_secs();
    let response = client
        .post(url.clone())
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("webhook-id", event_id)
        .header("webhook-timestamp", timestamp)
        .header(
            "webhook-signature",
            policy.signature(event_id, timestamp, body),
        )
        .body(body.to_string())
        .send()
        .await
        .map_err(|err| format!("the request failed: {err}"))?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(format!(
            "the endpoint answered HTTP {}",
            response.status().as_u16()
        ))
    }
}

#[cfg(test)]
#[path = "webhook_tests.rs"]
pub(crate) mod tests;
